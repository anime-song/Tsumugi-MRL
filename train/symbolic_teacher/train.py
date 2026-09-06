"""Train the symbolic encoder/RVQ from Tsumugi token caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train.config import TrainingConfig
from train.losses import SymbolicTeacherLoss
from train.symbolic import CachedSymbolicDataset, SymbolicMIDIDataset, collate_symbolic_sequences
from train.symbolic_teacher.model import SymbolicTeacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--token-dir",
        type=Path,
        default=Path("datasets/symbolic/tokens"),
        help="Directory of .pt caches written by train.symbolic_teacher.prepare_dataset.",
    )
    source.add_argument(
        "--midi-dir",
        type=Path,
        default=None,
        help="Tokenize MIDI on the fly instead of reading the .pt caches.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/symbolic_teacher"))
    parser.add_argument(
        "--crop-frames",
        type=int,
        default=750,
        help="Random window length in 25 Hz frames (750 = 30 s). Use 0 for whole songs.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade extra Transformer recomputation for lower activation memory.",
    )
    return parser


def _build_dataset(args: argparse.Namespace, config: TrainingConfig):
    crop_frames = args.crop_frames if args.crop_frames > 0 else None
    if args.midi_dir is not None:
        return SymbolicMIDIDataset(
            args.midi_dir,
            frame_rate=config.symbolic_frame_rate,
            crop_frames=crop_frames,
        )
    return CachedSymbolicDataset(
        args.token_dir,
        frame_rate=config.symbolic_frame_rate,
        crop_frames=crop_frames,
    )


def _save_checkpoint(
    teacher: SymbolicTeacher, config: TrainingConfig, output_dir: Path, epoch: int, loss: float
) -> None:
    checkpoint = {
        "state_dict": teacher.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "loss": loss,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / "last.pt")
    torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.crop_frames < 0:
        raise ValueError("crop_frames must be non-negative")
    torch.manual_seed(args.seed)
    config = TrainingConfig(gradient_checkpointing=args.gradient_checkpointing)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _build_dataset(args, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_symbolic_sequences,
    )
    teacher = SymbolicTeacher(config).to(device)
    criterion = SymbolicTeacherLoss(config.symbolic_reconstruction_weight)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.lr)

    source = args.midi_dir if args.midi_dir is not None else args.token_dir
    crop = f"{args.crop_frames} frames" if args.crop_frames > 0 else "whole song"
    print(f"source={source} files={len(dataset)} crop={crop} device={device} batch_size={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        teacher.train()
        total = 0.0
        for batch in loader:
            output = teacher(
                batch["token_ids"].to(device),
                anchor_positions=batch["anchor_positions"].to(device),
                token_instrument_ids=batch["token_instrument_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                padding_mask=batch["padding_mask"].to(device),
                position_ids=batch["position_ids"].to(device),
                frame_padding_mask=batch["frame_padding_mask"].to(device),
            )
            loss_values = criterion(
                output,
                batch["targets"].to(device),
                frame_padding_mask=batch["frame_padding_mask"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss_values["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss_values["loss_total"].detach())

        average = total / max(1, len(loader))
        _save_checkpoint(teacher, config, args.output_dir, epoch, average)
        print(f"epoch={epoch:03d} loss={average:.4f} saved={args.output_dir / 'last.pt'}")


if __name__ == "__main__":
    train(build_parser().parse_args())
