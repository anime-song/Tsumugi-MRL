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
from train.losses import PretrainingLoss
from train.mel_rvq.config import MelRVQConfig
from train.mel_rvq.model import MelRVQTokenizer
from train.pretraining import TsumugiMRLPretrainingModel

AUDIO_SETTINGS = {"d_model", "n_heads", "num_layers", "dim_feedforward", "dropout", "gradient_checkpointing"}
FRONTEND_SETTINGS = {"sample_rate", "audio_channels", "n_mels", "n_fft", "hop_length", "temporal_fold"}
ABLATION_MODES = ("mel_rvq", "symbolic_teacher", "contrastive")
DEFAULT_ABLATION = "contrastive"


def compile_audio_encoder(model: TsumugiMRLPretrainingModel) -> None:
    """Compile only the Transformer inside MaskedAudioEncoder."""

    model.audio_encoder.encoder.forward = torch.compile(
        model.audio_encoder.encoder.forward,
        mode="default",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    # Dataset and teacher checkpoints.
    parser.add_argument("--manifest", type=Path, default=Path("datasets/symbolic/manifest.json"))
    parser.add_argument("--mel-checkpoint", type=Path)
    parser.add_argument("--symbolic-checkpoint", type=Path)
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
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs, including completed epochs.")

    # Batch construction and masking.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-frames", type=int, default=750)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-span", type=int, default=10)

    # Optimization and runtime settings.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


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
    mel_path: Path,
    symbolic_path: Path | None,
    audio_settings: dict,
    use_symbolic: bool = True,
):
    # Reject settings that cannot be applied to the audio encoder.
    unknown = audio_settings.keys() - AUDIO_SETTINGS
    if unknown:
        raise ValueError(f"Unsupported audio settings: {sorted(unknown)}")

    # Load the acoustic teacher checkpoint. Mel-only ablations use its config
    # as the base; symbolic ablations additionally restore symbolic settings.
    mel = MelRVQTokenizer.from_checkpoint(mel_path)
    symbolic = None
    settings = asdict(mel.config)
    if use_symbolic:
        if symbolic_path is None:
            raise ValueError("This ablation requires --symbolic-checkpoint.")
        symbolic = torch.load(symbolic_path, map_location="cpu")
        # Start with the symbolic configuration, then restore the Mel frontend
        # and acoustic-head settings from the acoustic teacher.
        settings = dict(symbolic["config"])
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
        model.symbolic_teacher.load_state_dict(symbolic["state_dict"])
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
    with torch.no_grad():
        mel_features = mel_teacher.frontend(batch["audio"])
        acoustic_targets = mel_teacher.encode_features(mel_features)

    model_inputs = {
        "audio": batch["audio"],
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
        if args.mel_checkpoint is None:
            raise ValueError("A new run requires --mel-checkpoint.")
        if use_musical and args.symbolic_checkpoint is None:
            raise ValueError("This ablation requires --symbolic-checkpoint.")
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
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = PretrainingLoss(temperature=model.config.contrastive_temperature)
    dataset = PairedAudioDataset(args.manifest, model.config, args.crop_frames)
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
        loader_options.update(persistent_workers=True, prefetch_factor=2)
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

    # Prepare the checkpoint directory and report the data-loading setup.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"pairs={len(dataset)} batches={len(loader)} device={device} "
        f"amp={use_amp} amp_dtype={amp_dtype} compile_encoder={args.compile_encoder} "
        f"num_workers={args.num_workers}",
        flush=True,
    )

    # Train one epoch at a time so every completed epoch can be resumed.
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        totals = {}
        for batch in loader:
            # Move the complete collated batch to the model's device.
            batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}

            # Compute the masked objectives and update trainable parameters.
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
                    "num_workers",
                    "ablation",
                    "amp_dtype",
                    "compile_encoder",
                )
            },
        }
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch:04d}.pt")
        torch.save(checkpoint, args.output_dir / "last.pt")
        print(f"epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in averages.items()), flush=True)

    # Export the student without teacher and pretraining-only heads.
    model.export_audio_model().save_pretrained(args.output_dir / "audio_model")


if __name__ == "__main__":
    train(build_parser().parse_args())
