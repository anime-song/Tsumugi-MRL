"""Pretrain the audio model from paired audio and symbolic caches."""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from train.config import TrainingConfig
from train.data import PairedAudioDataset, collate_pretraining_windows
from train.loading import MEL_RVQ_SOURCE, SYMBOLIC_TEACHER_SOURCE, load_teacher
from train.losses import PretrainingLoss
from train.mel_rvq.config import MelRVQConfig
from train.mel_rvq.model import MelRVQTokenizer
from train.pretraining import TsumugiMRLPretrainingModel
from train.symbolic_teacher.model import SymbolicTeacher

AUDIO_SETTINGS = {"d_model", "n_heads", "num_layers", "dim_feedforward", "dropout", "gradient_checkpointing"}
FRONTEND_SETTINGS = {"sample_rate", "audio_channels", "n_mels", "n_fft", "hop_length", "temporal_fold"}
ABLATION_MODES = ("mel_rvq", "symbolic_teacher", "contrastive")
DEFAULT_ABLATION = "contrastive"


def _loss_metric_name(name: str) -> str:
    return "loss" if name == "loss_total" else name.removeprefix("loss_")


def compile_audio_encoder(model: TsumugiMRLPretrainingModel) -> None:
    """Compile only the Conformer inside MaskedAudioEncoder."""

    model.audio_encoder.encoder.forward = torch.compile(
        model.audio_encoder.encoder.forward,
        mode="default",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    # Dataset and teacher checkpoints.
    parser.add_argument("--manifest", type=Path, default=Path("datasets/symbolic/manifest.json"))
    parser.add_argument(
        "--mel-checkpoint",
        help=f"Training .pt, export directory, or Hub id. Defaults to {MEL_RVQ_SOURCE}.",
    )
    parser.add_argument(
        "--symbolic-checkpoint",
        help=f"Training .pt, export directory, or Hub id. Defaults to {SYMBOLIC_TEACHER_SOURCE}.",
    )
    parser.add_argument("--config", type=Path, help="JSON audio encoder settings for a new run.")
    parser.add_argument("--resume", type=Path, help="Resume an epoch checkpoint, including both teachers.")
    parser.add_argument(
        "--ablation",
        choices=ABLATION_MODES,
        default=None,
        help=(
            "Pretraining objectives: mel_rvq (acoustic only), symbolic_teacher "
            "(+symbolic code prediction), or contrastive (+Audio--MIDI contrastive). "
            "Defaults to contrastive."
        ),
    )

    # Run length and output location.
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pretraining"))
    parser.add_argument("--epochs", type=int, default=800, help="Total number of epochs, including completed epochs.")

    # Batch construction and masking.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-frames", type=int, default=750)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-span", type=int, default=18)

    # Optimization and runtime settings. The defaults follow MuQ's fairseq
    # pretraining recipe (Adam 5e-4, betas 0.9/0.98, eps 1e-6, weight decay
    # 0.01, clip 10, linear warmup into a polynomial decay).
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate reached at the end of warmup.")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.98), metavar=("BETA1", "BETA2"))
    parser.add_argument("--adam-eps", type=float, default=1e-6)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.08,
        help=(
            "Warmup length as a fraction of the whole run. MuQ warms up for 32,000 of its "
            "400,000 updates; a fraction transfers that shape to a run of any length."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Absolute warmup length in updates; overrides --warmup-ratio when positive.",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=0,
        help="Updates the decay spans; defaults to the whole requested run (epochs x batches).",
    )
    parser.add_argument("--lr-end", type=float, default=0.0, help="Learning rate the decay ends at.")
    parser.add_argument("--lr-power", type=float, default=1.0, help="Decay exponent; 1.0 decays linearly.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; use 4-8 when the CPU/SSD can keep the GPU fed.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Keep epoch_XXXX.pt every N epochs; last.pt is always written.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="CUDA AMP dtype; auto prefers bfloat16 when supported.",
    )
    parser.add_argument(
        "--compile-encoder",
        action="store_true",
        help="Compile only MaskedAudioEncoder.encoder with torch.compile(mode='default').",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="tsumugi-mrl-pretraining")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def polynomial_decay_lr(
    update: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    end_lr: float = 0.0,
    power: float = 1.0,
) -> float:
    """Return the learning rate for a one-based update number.

    This reproduces fairseq's ``polynomial_decay`` schedule, which is what MuQ
    pretrains with: the rate rises linearly from zero to ``base_lr`` over the
    warmup, then falls to ``end_lr`` over the remaining updates. A masked
    prediction loss is unstable while the encoder still produces noise, so the
    warmup matters more here than the exact shape of the decay.
    """

    if warmup_steps > 0 and update <= warmup_steps:
        return base_lr * update / warmup_steps
    if update >= total_steps:
        return end_lr
    remaining = 1 - (update - warmup_steps) / max(total_steps - warmup_steps, 1)
    return (base_lr - end_lr) * remaining**power + end_lr


def make_span_mask(padding_mask: Tensor, ratio: float, span: int) -> Tensor:
    """Mask randomly selected contiguous spans of valid audio tokens [B, T]."""
    if not 0 < ratio < 1 or span <= 0:
        raise ValueError("mask-ratio must be between 0 and 1 and mask-span must be positive.")

    # Build a separate mask for each example without touching padded tokens.
    mask = torch.zeros_like(padding_mask)
    # One host transfer is enough; doing .item() once per row synchronizes a
    # CUDA stream repeatedly before the actual encoder work starts.
    lengths = (~padding_mask).sum(dim=1).tolist()
    for row, length in enumerate(lengths):
        target = max(1, round(length * ratio))

        # Shuffle non-overlapping spans and trim the last one to the budget.
        remaining = target
        for block in torch.randperm((length + span - 1) // span).tolist():
            start = block * span
            count = min(span, length - start, remaining)
            mask[row, start : start + count] = True
            remaining -= count
            if remaining == 0:
                break
    return mask


def load_teachers(
    mel_path: str | Path | None = None,
    symbolic_path: str | Path | None = None,
    audio_settings: dict | None = None,
    use_symbolic: bool = True,
):
    """Build the pretraining model on top of both frozen teachers.

    Either teacher may come from a local training ``.pt``, a ``save_pretrained``
    directory, or the Hub. ``None`` selects the published release, so a fresh
    checkout can pretrain without training the teachers first.
    """

    audio_settings = audio_settings or {}
    # Reject settings that cannot be applied to the audio encoder.
    unknown = audio_settings.keys() - AUDIO_SETTINGS
    if unknown:
        raise ValueError(f"Unsupported audio settings: {sorted(unknown)}")

    # Load the acoustic teacher checkpoint. Mel-only ablations use its config
    # as the base; symbolic ablations additionally restore symbolic settings.
    mel = load_teacher(MelRVQTokenizer, mel_path, MEL_RVQ_SOURCE)
    symbolic = None
    settings = asdict(mel.config)
    if use_symbolic:
        symbolic = load_teacher(SymbolicTeacher, symbolic_path, SYMBOLIC_TEACHER_SOURCE)
        # Start with the symbolic configuration, then restore the Mel frontend
        # and acoustic-head settings from the acoustic teacher.
        settings = asdict(symbolic.config)
    for field in fields(MelRVQConfig):
        if field.name in FRONTEND_SETTINGS or field.name.startswith(("acoustic_", "rvq_")):
            settings[field.name] = getattr(mel.config, field.name)

    # Apply optional audio-encoder overrides for a new run.
    settings.update(audio_settings)
    config = TrainingConfig(**settings)

    # Both teachers must produce targets at the same frame rate.
    if use_symbolic and config.encoder_frame_rate != config.symbolic_frame_rate:
        raise ValueError("Mel and symbolic teacher frame rates must match.")

    # Assemble the pretraining model and restore the symbolic teacher when it
    # participates in the selected objective set.
    model = TsumugiMRLPretrainingModel(config)
    if symbolic is not None:
        model.symbolic_teacher.load_state_dict(symbolic.state_dict())
    model.set_mel_stats(*mel.mel_stats)
    return model, mel


def _ablation_flags(ablation: str) -> tuple[bool, bool]:
    """Return whether symbolic targets and contrastive learning are active."""

    if ablation not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {ablation!r}")
    return ablation != "mel_rvq", ablation == "contrastive"


def pretraining_losses(
    model,
    mel_teacher,
    batch,
    criterion,
    mask_ratio,
    mask_span,
    ablation=DEFAULT_ABLATION,
):
    use_musical, use_contrastive = _ablation_flags(ablation)
    # Mask only valid audio frames; padding remains excluded from the target.
    audio_mask = make_span_mask(batch["audio_padding_mask"], mask_ratio, mask_span)

    # The acoustic teacher and student share this normalized Mel frontend.
    # A disk feature cache can provide the normalized Mel tensor directly,
    # avoiding WAV decoding and STFT work in every training epoch.
    with torch.no_grad():
        mel_features = batch.get("mel_features")
        if mel_features is None:
            mel_features = mel_teacher.frontend(batch["audio"])
        acoustic_targets = mel_teacher.encode_features(mel_features)

    model_inputs = {
        "audio": batch.get("audio"),
        "audio_mask": audio_mask,
        "audio_padding_mask": batch["audio_padding_mask"],
        "audio_mel": mel_features,
    }
    if use_musical:
        # Keep the symbolic encoder/RVQ fixed while using its frame codes as
        # targets. The symbolic projection remains trainable only for the
        # contrastive ablation.
        model_inputs.update(
            symbolic_token_ids=batch["symbolic_token_ids"],
            symbolic_token_instrument_ids=batch["symbolic_token_instrument_ids"],
            symbolic_token_type_ids=batch["symbolic_token_type_ids"],
            symbolic_padding_mask=batch["symbolic_padding_mask"],
            symbolic_position_ids=batch["symbolic_position_ids"],
            symbolic_anchor_positions=batch["symbolic_anchor_positions"],
            symbolic_frame_padding_mask=batch["symbolic_frame_padding_mask"],
        )
    output = model(**model_inputs)

    musical_targets = None
    if use_musical:
        # Select symbolic targets aligned to the audio frames in each crop.
        indices = batch["symbolic_frame_indices"].unsqueeze(-1).expand(-1, -1, output.symbolic_codes.size(-1))
        musical_targets = output.symbolic_codes.gather(1, indices)

    # Exclude padded audio frames from every prediction loss.
    valid = ~batch["audio_padding_mask"]

    # Combine acoustic, symbolic, and contrastive objectives.
    return criterion(
        output,
        acoustic_targets,
        musical_targets,
        acoustic_mask=audio_mask,
        musical_mask=audio_mask,
        acoustic_valid_mask=valid,
        musical_valid_mask=valid,
        use_musical=use_musical,
        use_contrastive=use_contrastive,
    )


def train(args: argparse.Namespace) -> None:
    # Validate arguments before constructing models or loading data.
    if args.epochs <= 0 or args.batch_size < 2 or args.crop_frames <= 0:
        raise ValueError("epochs/crop-frames must be positive and batch-size must be at least 2.")
    if args.lr <= 0 or args.grad_clip <= 0 or args.log_interval <= 0 or args.num_workers < 0:
        raise ValueError("lr, grad-clip, and log-interval must be positive; num-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch-factor must be positive.")
    if args.weight_decay < 0 or args.adam_eps <= 0 or not all(0 <= beta < 1 for beta in args.adam_betas):
        raise ValueError("weight-decay must be non-negative, adam-eps positive, and adam-betas in [0, 1).")
    if not 0 <= args.warmup_ratio < 1 or args.warmup_steps < 0 or args.decay_steps < 0:
        raise ValueError("warmup-ratio must be in [0, 1) and warmup/decay-steps non-negative.")
    if args.lr_end < 0 or args.lr_end > args.lr or args.lr_power <= 0:
        raise ValueError("lr-end must be between zero and --lr, and lr-power must be positive.")
    if args.save_interval <= 0:
        raise ValueError("save-interval must be positive.")
    if not 0 < args.mask_ratio < 1 or args.mask_span <= 0:
        raise ValueError("mask-ratio must be between 0 and 1 and mask-span must be positive.")
    if args.resume and any((args.config, args.mel_checkpoint, args.symbolic_checkpoint)):
        raise ValueError("--resume restores config and teachers; omit --config and teacher paths.")
    requested_ablation = getattr(args, "ablation", None)

    # Seed the run and select the requested accelerator.
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Restore a complete run, or build a new model from the teacher checkpoints.
    restored = None
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu")
        restored_ablation = restored["training_settings"].get("ablation", DEFAULT_ABLATION)
        if requested_ablation is not None and requested_ablation != restored_ablation:
            raise ValueError("--resume restores the ablation mode; omit --ablation or use the saved mode.")
        ablation = restored_ablation
        _ablation_flags(ablation)
        model = TsumugiMRLPretrainingModel(TrainingConfig(**restored["config"]))
        model.load_state_dict(restored["state_dict"])
        mel = MelRVQTokenizer(MelRVQConfig(**restored["mel_config"]))
        mel.load_state_dict(restored["mel_state_dict"])
        for name, value in restored["training_settings"].items():
            if hasattr(args, name):
                setattr(args, name, value)
    else:
        ablation = requested_ablation or DEFAULT_ABLATION
        use_musical, _ = _ablation_flags(ablation)
        # Either teacher may be omitted; load_teachers then uses the release.
        settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        model, mel = load_teachers(
            args.mel_checkpoint,
            args.symbolic_checkpoint,
            settings,
            use_symbolic=use_musical,
        )
    args.ablation = ablation

    # Freeze both teachers. Only the symbolic projection is trainable when it
    # supplies the contrastive embedding.
    model.to(device)
    mel.to(device).eval().requires_grad_(False)
    model.symbolic_teacher.freeze(train_projection=ablation == "contrastive")
    if args.compile_encoder:
        compile_audio_encoder(model)

    use_amp = device.type == "cuda"
    amp_dtype = torch.float32
    if use_amp:
        if args.amp_dtype == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        elif args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    if restored and restored.get("amp_scaler") is not None:
        scaler.load_state_dict(restored["amp_scaler"])

    # Create the optimizer, loss function, and paired audio/MIDI data loader.
    # fairseq's "adam" decays the weights separately from the gradient, so
    # torch's AdamW is what reproduces MuQ's optimizer rather than torch's Adam.
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        betas=tuple(args.adam_betas),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    criterion = PretrainingLoss(temperature=model.config.contrastive_temperature)
    dataset = PairedAudioDataset(args.manifest, model.config, args.crop_frames)
    cache_mode = "mel_features" if dataset.pairs[0].mel_path is not None else "waveform"
    if len(dataset) < args.batch_size:
        raise ValueError("Not enough successful pairs for one full batch; lower --batch-size (minimum 2).")
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.num_workers,
        "collate_fn": collate_pretraining_windows,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    loader = DataLoader(dataset, **loader_options)

    # Restore optimizer and random state after rebuilding the training objects.
    start_epoch, step = 0, 0
    if restored:
        optimizer.load_state_dict(restored["optimizer"])
        start_epoch, step = restored["epoch"], restored["step"]
        torch.set_rng_state(restored["rng_state"])
        if device.type == "cuda" and restored["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng_state"])
    if args.epochs <= start_epoch:
        raise ValueError("--epochs must exceed the number of completed epochs.")

    # By default the decay spans the whole requested run, so extending --epochs
    # on a resume stretches the schedule instead of leaving the rate at
    # --lr-end; --decay-steps pins it when a run should keep its original
    # horizon. The warmup is resolved to updates once and then carried in the
    # checkpoint, since a run that is extended has already finished warming up.
    total_steps = args.decay_steps or args.epochs * len(loader)
    if args.warmup_steps <= 0:
        args.warmup_steps = round(total_steps * args.warmup_ratio)
    if args.warmup_steps >= total_steps:
        raise ValueError("The warmup covers the whole run; lower --warmup-steps or --warmup-ratio.")

    # Prepare the checkpoint directory and report the data-loading setup.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requires the optional 'wandb' package. Install it with: uv sync --extra train"
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                **asdict(model.config),
                "manifest": str(args.manifest),
                "mel_checkpoint": str(args.mel_checkpoint) if args.mel_checkpoint else None,
                "symbolic_checkpoint": str(args.symbolic_checkpoint) if args.symbolic_checkpoint else None,
                "ablation": ablation,
                "pairs": len(dataset),
                "cache_mode": cache_mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "crop_frames": args.crop_frames,
                "mask_ratio": args.mask_ratio,
                "mask_span": args.mask_span,
                "learning_rate": args.lr,
                "grad_clip": args.grad_clip,
                "weight_decay": args.weight_decay,
                "adam_betas": tuple(args.adam_betas),
                "adam_eps": args.adam_eps,
                "warmup_steps": args.warmup_steps,
                "total_steps": total_steps,
                "lr_end": args.lr_end,
                "lr_power": args.lr_power,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "save_interval": args.save_interval,
                "device": str(device),
                "amp": use_amp,
                "amp_dtype": str(amp_dtype),
                "compile_encoder": args.compile_encoder,
                "seed": args.seed,
            },
        )

    print(
        f"pairs={len(dataset)} batches={len(loader)} device={device} "
        f"amp={use_amp} amp_dtype={amp_dtype} compile_encoder={args.compile_encoder} "
        f"num_workers={args.num_workers} prefetch_factor={args.prefetch_factor} cache={cache_mode}\n"
        f"lr={args.lr} warmup={args.warmup_steps}/{total_steps} updates "
        f"({args.warmup_steps / total_steps:.1%}) weight_decay={args.weight_decay} clip={args.grad_clip}",
        flush=True,
    )

    # Train one epoch at a time so every completed epoch can be resumed.
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            totals = {}
            for batch in loader:
                # Move the complete collated batch to the model's device.
                batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}

                # Compute the masked objectives and update trainable parameters.
                # The rate is derived from the update counter rather than held
                # in a scheduler object, so a resume picks the schedule back up
                # from the restored step without extra state. fairseq updates
                # the rate after each step, so an update sees the rate computed
                # from the updates that came before it.
                for group in optimizer.param_groups:
                    group["lr"] = polynomial_decay_lr(
                        max(step, 1),
                        args.lr,
                        args.warmup_steps,
                        total_steps,
                        args.lr_end,
                        args.lr_power,
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    losses = pretraining_losses(
                        model,
                        mel,
                        batch,
                        criterion,
                        args.mask_ratio,
                        args.mask_span,
                        ablation=ablation,
                    )
                if not torch.isfinite(losses["loss_total"]):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}.")
                if scaler.is_enabled():
                    scaler.scale(losses["loss_total"]).backward()
                else:
                    losses["loss_total"].backward()

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                step += 1

                # Accumulate epoch metrics and print periodic progress.
                loss_values = dict(zip(losses, torch.stack(tuple(losses.values())).detach().cpu().tolist()))
                for name, value in loss_values.items():
                    totals[name] = totals.get(name, 0.0) + value
                if step % args.log_interval == 0:
                    print(
                        f"epoch={epoch} step={step} " + " ".join(f"{k}={v:.4f}" for k, v in loss_values.items()),
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                **{f"train/{_loss_metric_name(name)}": value for name, value in loss_values.items()},
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                            },
                            step=step,
                        )

            # Average metrics and save all state needed for an exact resume.
            averages = {key: value / len(loader) for key, value in totals.items()}
            checkpoint = {
                "config": asdict(model.config),
                "state_dict": model.state_dict(),
                "mel_config": asdict(mel.config),
                "mel_state_dict": mel.state_dict(),
                "optimizer": optimizer.state_dict(),
                "amp_scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "losses": averages,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "training_settings": {
                    name: getattr(args, name)
                    for name in (
                        "batch_size",
                        "crop_frames",
                        "mask_ratio",
                        "mask_span",
                        "lr",
                        "grad_clip",
                        "weight_decay",
                        "adam_betas",
                        "adam_eps",
                        "warmup_steps",
                        "lr_end",
                        "lr_power",
                        "num_workers",
                        "prefetch_factor",
                        "ablation",
                        "amp_dtype",
                        "compile_encoder",
                    )
                },
            }
            # last.pt always allows an exact resume; the numbered copies are
            # kept on the requested interval and for the final epoch.
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                torch.save(checkpoint, args.output_dir / f"epoch_{epoch:04d}.pt")
            torch.save(checkpoint, args.output_dir / "last.pt")
            print(f"epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in averages.items()), flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"epoch/{_loss_metric_name(name)}": value for name, value in averages.items()},
                        "epoch": epoch,
                    },
                    step=step,
                )

        # Export the student without teacher and pretraining-only heads.
        model.export_audio_model().save_pretrained(args.output_dir / "audio_model")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    train(build_parser().parse_args())
