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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/symbolic/manifest.json"))
    parser.add_argument("--mel-checkpoint", type=Path)
    parser.add_argument("--symbolic-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="JSON audio encoder settings for a new run.")
    parser.add_argument("--resume", type=Path, help="Resume an epoch checkpoint, including both teachers.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pretraining"))
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs, including completed epochs.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-frames", type=int, default=750)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-span", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def make_span_mask(padding_mask: Tensor, ratio: float, span: int) -> Tensor:
    """Mask randomly selected contiguous spans of valid audio tokens [B, T]."""
    if not 0 < ratio < 1 or span <= 0:
        raise ValueError("mask-ratio must be between 0 and 1 and mask-span must be positive.")
    mask = torch.zeros_like(padding_mask)
    for row in range(mask.size(0)):
        length = int((~padding_mask[row]).sum().item())
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


def load_teachers(mel_path: Path, symbolic_path: Path, audio_settings: dict):
    unknown = audio_settings.keys() - AUDIO_SETTINGS
    if unknown:
        raise ValueError(f"Unsupported audio settings: {sorted(unknown)}")
    mel = MelRVQTokenizer.from_checkpoint(mel_path)
    symbolic = torch.load(symbolic_path, map_location="cpu")
    settings = dict(symbolic["config"])
    # The Mel checkpoint defines the frontend and acoustic prediction heads;
    # the symbolic checkpoint defines the symbolic architecture and projection.
    for field in fields(MelRVQConfig):
        if field.name in FRONTEND_SETTINGS or field.name.startswith(("acoustic_", "rvq_")):
            settings[field.name] = getattr(mel.config, field.name)
    settings.update(audio_settings)
    config = TrainingConfig(**settings)
    if config.encoder_frame_rate != config.symbolic_frame_rate:
        raise ValueError("Mel and symbolic teacher frame rates must match.")
    model = TsumugiMRLPretrainingModel(config)
    model.symbolic_teacher.load_state_dict(symbolic["state_dict"])
    model.set_mel_stats(*mel.mel_stats)
    return model, mel


def pretraining_losses(model, mel_teacher, batch, criterion, mask_ratio, mask_span):
    audio_mask = make_span_mask(batch["audio_padding_mask"], mask_ratio, mask_span)
    with torch.no_grad():
        acoustic_targets = mel_teacher.encode(batch["audio"])
    # Keep gradients through the teacher's contrastive projection. Its encoder,
    # RVQ, and decoder are frozen by freeze(), not by a surrounding no_grad().
    output = model(
        batch["audio"],
        audio_mask=audio_mask,
        audio_padding_mask=batch["audio_padding_mask"],
        symbolic_token_ids=batch["symbolic_token_ids"],
        symbolic_token_instrument_ids=batch["symbolic_token_instrument_ids"],
        symbolic_token_type_ids=batch["symbolic_token_type_ids"],
        symbolic_padding_mask=batch["symbolic_padding_mask"],
        symbolic_position_ids=batch["symbolic_position_ids"],
        symbolic_anchor_positions=batch["symbolic_anchor_positions"],
        symbolic_frame_padding_mask=batch["symbolic_frame_padding_mask"],
    )
    indices = batch["symbolic_frame_indices"].unsqueeze(-1).expand(-1, -1, output.symbolic_codes.size(-1))
    musical_targets = output.symbolic_codes.gather(1, indices)
    valid = ~batch["audio_padding_mask"]
    return criterion(
        output,
        acoustic_targets,
        musical_targets,
        acoustic_mask=audio_mask,
        musical_mask=audio_mask,
        acoustic_valid_mask=valid,
        musical_valid_mask=valid,
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size < 2 or args.crop_frames <= 0:
        raise ValueError("epochs/crop-frames must be positive and batch-size must be at least 2.")
    if args.lr <= 0 or args.grad_clip <= 0 or args.log_interval <= 0 or args.num_workers < 0:
        raise ValueError("lr, grad-clip, and log-interval must be positive; num-workers must be non-negative.")
    if not 0 < args.mask_ratio < 1 or args.mask_span <= 0:
        raise ValueError("mask-ratio must be between 0 and 1 and mask-span must be positive.")
    if args.resume and any((args.config, args.mel_checkpoint, args.symbolic_checkpoint)):
        raise ValueError("--resume restores config and teachers; omit --config and teacher paths.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    restored = None
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu")
        model = TsumugiMRLPretrainingModel(TrainingConfig(**restored["config"]))
        model.load_state_dict(restored["state_dict"])
        mel = MelRVQTokenizer(MelRVQConfig(**restored["mel_config"]))
        mel.load_state_dict(restored["mel_state_dict"])
        for name, value in restored["training_settings"].items():
            setattr(args, name, value)
    else:
        if args.mel_checkpoint is None or args.symbolic_checkpoint is None:
            raise ValueError("A new run requires --mel-checkpoint and --symbolic-checkpoint.")
        settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        model, mel = load_teachers(args.mel_checkpoint, args.symbolic_checkpoint, settings)
    model.to(device)
    mel.to(device).eval().requires_grad_(False)
    model.symbolic_teacher.freeze()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = PretrainingLoss(temperature=model.config.contrastive_temperature)
    dataset = PairedAudioDataset(args.manifest, model.config, args.crop_frames)
    if len(dataset) < args.batch_size:
        raise ValueError("Not enough successful pairs for one full batch; lower --batch-size (minimum 2).")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_pretraining_windows,
    )
    start_epoch, step = 0, 0
    if restored:
        optimizer.load_state_dict(restored["optimizer"])
        start_epoch, step = restored["epoch"], restored["step"]
        torch.set_rng_state(restored["rng_state"])
        if device.type == "cuda" and restored["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng_state"])
    if args.epochs <= start_epoch:
        raise ValueError("--epochs must exceed the number of completed epochs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"pairs={len(dataset)} batches={len(loader)} device={device}", flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        totals = {}
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses = pretraining_losses(model, mel, batch, criterion, args.mask_ratio, args.mask_span)
            if not torch.isfinite(losses["loss_total"]):
                raise RuntimeError(f"Non-finite loss at step {step + 1}.")
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item()
            if step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={step} " + " ".join(f"{k}={v.detach().item():.4f}" for k, v in losses.items()),
                    flush=True,
                )
        averages = {key: value / len(loader) for key, value in totals.items()}
        # Epoch checkpoints include teacher weights and optimizer/RNG state so
        # resuming does not require the original teacher checkpoint files.
        checkpoint = {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "mel_config": asdict(mel.config),
            "mel_state_dict": mel.state_dict(),
            "optimizer": optimizer.state_dict(),
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
                )
            },
        }
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch:04d}.pt")
        torch.save(checkpoint, args.output_dir / "last.pt")
        print(f"epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in averages.items()), flush=True)
    model.export_audio_model().save_pretrained(args.output_dir / "audio_model")


if __name__ == "__main__":
    train(build_parser().parse_args())
