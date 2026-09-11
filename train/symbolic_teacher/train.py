"""Train the symbolic encoder/RVQ from Tsumugi token caches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler, Subset

from train.config import TrainingConfig
from train.losses import SymbolicTeacherLoss
from train.symbolic import CachedSymbolicDataset, SymbolicMIDIDataset, collate_symbolic_sequences
from train.symbolic_teacher.model import SymbolicTeacher

_SYMBOLIC_BINARY_STAT_NAMES = ("note_onset", "note_offset")
_SYMBOLIC_CATEGORICAL_STAT_NAMES = ("chord", "meter")


def _loss_metric_name(name: str) -> str:
    return "loss" if name == "loss_total" else name.removeprefix("loss_")


def _f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def _macro_f1_from_counts(counts: dict[str, Tensor]) -> float:
    support = counts["support"]
    active = support > 0
    if not bool(active.any()):
        return 0.0
    true_positive = counts["true_positive"].to(dtype=torch.float64)
    false_positive = counts["predicted"].to(dtype=torch.float64) - true_positive
    false_negative = support.to(dtype=torch.float64) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = 2.0 * true_positive / denominator.clamp_min(1.0)
    return float(f1[active].mean().item())


def _perplexity_from_counts(counts: Tensor) -> float:
    totals = counts.sum(dim=1)
    active = totals > 0
    if not bool(active.any()):
        return 0.0
    probabilities = counts / totals.clamp_min(1.0).unsqueeze(1)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return float(entropy[active].exp().mean().item())


def _update_binary_counts(
    counts: dict[str, list[int]], name: str, logits: Tensor, targets: Tensor, valid_frame: Tensor
) -> None:
    active = valid_frame
    while active.ndim < targets.ndim:
        active = active.unsqueeze(-1)
    active = active.expand_as(targets)
    predicted = logits >= 0
    positive = targets > 0.5
    counts[name][0] += int((predicted & positive & active).sum().item())
    counts[name][1] += int((predicted & ~positive & active).sum().item())
    counts[name][2] += int((~predicted & positive & active).sum().item())


def _update_categorical_counts(
    counts: dict[str, dict[str, Tensor]], name: str, logits: Tensor, targets: Tensor, valid_frame: Tensor
) -> None:
    active = valid_frame & (targets != -100)
    predicted = logits.argmax(dim=-1)
    target_values = targets[active].long().cpu()
    predicted_values = predicted[active].long().cpu()
    classes = logits.size(-1)
    if name not in counts:
        counts[name] = {
            "true_positive": torch.zeros(classes, dtype=torch.long),
            "predicted": torch.zeros(classes, dtype=torch.long),
            "support": torch.zeros(classes, dtype=torch.long),
        }
    counts[name]["true_positive"] += torch.bincount(target_values[target_values == predicted_values], minlength=classes)
    counts[name]["predicted"] += torch.bincount(predicted_values, minlength=classes)
    counts[name]["support"] += torch.bincount(target_values, minlength=classes)


def _split_dataset(dataset, val_ratio: float, seed: int):
    if not 0 <= val_ratio < 1:
        raise ValueError("val-ratio must be in [0, 1).")
    if val_ratio == 0:
        return dataset, None
    if len(dataset) < 2:
        raise ValueError("A validation split requires at least two symbolic files.")

    validation_size = min(len(dataset) - 1, max(1, round(len(dataset) * val_ratio)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, indices[validation_size:]), Subset(dataset, indices[:validation_size])


def _compute_symbolic_loss_statistics(dataset, indices: list[int], config: TrainingConfig, seed: int):
    """Count train-window labels for sparse binary and long-tail categorical heads."""

    binary_statistics = {
        name: {
            "positive": torch.zeros((), dtype=torch.float64),
            "negative": torch.zeros((), dtype=torch.float64),
        }
        for name in _SYMBOLIC_BINARY_STAT_NAMES
    }
    categorical_statistics = {
        name: torch.zeros(
            getattr(config, f"symbolic_{name}_classes"),
            dtype=torch.float64,
        )
        for name in _SYMBOLIC_CATEGORICAL_STAT_NAMES
    }
    random_state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        for index in indices:
            targets = dataset[index].frame_targets
            for name in _SYMBOLIC_BINARY_STAT_NAMES:
                values = getattr(targets, name).to(dtype=torch.float64)
                binary_statistics[name]["positive"] += values.sum()
                binary_statistics[name]["negative"] += values.numel() - values.sum()
            for name in _SYMBOLIC_CATEGORICAL_STAT_NAMES:
                values = getattr(targets, name)
                valid = values >= 0
                if valid.any():
                    categorical_statistics[name] += torch.bincount(
                        values[valid].long(), minlength=categorical_statistics[name].numel()
                    ).to(dtype=torch.float64)
    finally:
        torch.set_rng_state(random_state)
    return {
        "dataset_length": len(dataset),
        "indices": list(indices),
        "crop_frames": getattr(dataset, "crop_frames", None),
        "binary": binary_statistics,
        "categorical": categorical_statistics,
    }


def _load_or_compute_symbolic_loss_statistics(
    dataset,
    train_indices: list[int],
    config: TrainingConfig,
    seed: int,
    path: Path,
    recompute: bool,
):
    if not recompute and path.is_file():
        statistics = torch.load(path, map_location="cpu")
        if (
            statistics.get("dataset_length") == len(dataset)
            and statistics.get("indices") == list(train_indices)
            and statistics.get("crop_frames") == getattr(dataset, "crop_frames", None)
        ):
            return statistics

    print(f"computing symbolic loss statistics: files={len(train_indices)} path={path}")
    statistics = _compute_symbolic_loss_statistics(dataset, train_indices, config, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(statistics, path)
    return statistics


def _build_symbolic_loss_weights(
    statistics,
    *,
    beat_pos_weight: float,
    downbeat_pos_weight: float,
    max_note_pos_weight: float = 50.0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    binary_pos_weights: dict[str, Tensor] = {
        "beat": torch.tensor(float(beat_pos_weight)),
        "downbeat": torch.tensor(float(downbeat_pos_weight)),
    }
    for name in _SYMBOLIC_BINARY_STAT_NAMES:
        positive = statistics["binary"][name]["positive"]
        negative = statistics["binary"][name]["negative"]
        # Cap rare-event weights as in the AMT pair-gate loss to avoid unstable updates.
        binary_pos_weights[name] = (negative / positive.clamp_min(1.0)).clamp(1.0, max_note_pos_weight).float()
    categorical_class_counts = {name: counts.float() for name, counts in statistics["categorical"].items()}
    return binary_pos_weights, categorical_class_counts


@torch.no_grad()
def _evaluate(
    teacher: SymbolicTeacher,
    loader: DataLoader,
    criterion: SymbolicTeacherLoss,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    validation_seed: int,
    codebook_size: int,
) -> dict[str, float]:
    """Evaluate fixed file splits without backpropagating validation graphs."""

    teacher.eval()
    random_state = torch.get_rng_state()
    torch.manual_seed(validation_seed)
    totals: dict[str, float] = {}
    binary_counts = {name: [0, 0, 0] for name in ("note_onset", "note_offset", "beat", "downbeat")}
    categorical_counts: dict[str, dict[str, Tensor]] = {}
    rvq_counts: Tensor | None = None
    try:
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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

            for name, value in loss_values.items():
                totals[name] = totals.get(name, 0.0) + value.item()

            valid_frame = ~batch["frame_padding_mask"].to(device).bool()
            targets = batch["targets"].to(device)
            for name in binary_counts:
                _update_binary_counts(
                    binary_counts, name, output.reconstruction[name], getattr(targets, name), valid_frame
                )
            _update_categorical_counts(
                categorical_counts,
                "chord",
                output.reconstruction["chord"],
                targets.chord,
                valid_frame,
            )

            codes = output.codes.detach()[valid_frame].cpu()
            if rvq_counts is None:
                rvq_counts = torch.zeros((codes.size(-1), codebook_size), dtype=torch.float64)
            for codebook in range(codes.size(-1)):
                rvq_counts[codebook] += torch.bincount(codes[:, codebook], minlength=codebook_size).to(
                    dtype=torch.float64
                )
    finally:
        # Validation crops use a fixed RNG stream and must not perturb training.
        torch.set_rng_state(random_state)

    denominator = max(1, len(loader))
    metrics = {f"val/{_loss_metric_name(name)}": value / denominator for name, value in totals.items()}
    for name, (true_positive, false_positive, false_negative) in binary_counts.items():
        metrics[f"val/{name}_f1"] = _f1_from_counts(true_positive, false_positive, false_negative)
    metrics["val/chord_macro_f1"] = _macro_f1_from_counts(categorical_counts["chord"])
    metrics["val/rvq_perplexity"] = _perplexity_from_counts(rvq_counts) if rvq_counts is not None else 0.0
    return metrics


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
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of symbolic files held out for validation; use 0 to disable.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--beat-pos-weight", type=float, default=5.0)
    parser.add_argument("--downbeat-pos-weight", type=float, default=20.0)
    parser.add_argument(
        "--max-note-pos-weight",
        type=float,
        default=10.0,
        help="Upper bound for the data-derived note onset/offset positive weights.",
    )
    parser.add_argument(
        "--balanced-softmax-tau",
        type=float,
        default=0.3,
        help="Log-frequency correction strength for chord and meter losses; use 0 to disable.",
    )
    parser.add_argument(
        "--loss-stats-path",
        type=Path,
        default=None,
        help="Cached train-label statistics path; defaults to <output-dir>/symbolic_loss_stats.pt.",
    )
    parser.add_argument(
        "--recompute-loss-stats",
        action="store_true",
        help="Recount train labels even when a cached loss-statistics file exists.",
    )
    parser.add_argument(
        "--musical-codebooks",
        type=int,
        default=8,
        help="Residual stages in the symbolic RVQ; each stage adds about 9 bits per frame.",
    )
    parser.add_argument("--musical-codebook-dim", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--length-buckets",
        action="store_true",
        help="Batch items of similar length together to cut padding in the event attention.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="CUDA AMP dtype; auto prefers native bfloat16 and falls back to float16.",
    )
    parser.add_argument(
        "--compile-encoder",
        action="store_true",
        help="Compile only the SymbolicEncoder Transformer with torch.compile.",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode for --compile-encoder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="tsumugi-mrl-symbolic-teacher")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade extra Transformer recomputation for lower activation memory.",
    )
    return parser


class LengthBucketSampler(Sampler[list[int]]):
    """Group items of similar length into batches.

    Event sequences vary from a few hundred to a few thousand tokens, and
    ``collate_symbolic_sequences`` pads every batch to its longest member. With
    random batches roughly a fifth of the quadratic attention work is spent on
    padding. Sorting by length first removes almost all of it; the batches
    themselves are still shuffled, so the model does not see them in a fixed
    order.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for the given epoch, as a distributed sampler would."""

        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.argsort(
            torch.tensor(self.lengths, dtype=torch.long)
            # Break ties randomly so equal-length items are not always paired.
            + torch.randint(0, 2, (len(self.lengths),), generator=generator)
        ).tolist()
        batches = [order[start : start + self.batch_size] for start in range(0, len(order), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            batches = [batches[index] for index in torch.randperm(len(batches), generator=generator).tolist()]
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        count = len(self.lengths) // self.batch_size
        return count if self.drop_last else (len(self.lengths) + self.batch_size - 1) // self.batch_size


def _token_count(samples) -> int:
    """Collate one item into its token count.

    A module-level function rather than a lambda: DataLoader workers are
    spawned on Windows and have to pickle whatever they are given.
    """

    return int(samples[0].token_ids.numel())


def _sequence_lengths(dataset, num_workers: int, cache_path: Path, crop_frames: int | None) -> list[int]:
    """Token counts per item, cached on disk.

    The crop is random, so a length is an estimate of what an epoch will see
    rather than an exact count. That is enough to keep similar items together.
    """

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("items") == len(dataset) and payload.get("crop_frames") == crop_frames:
            return payload["lengths"]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_token_count,
    )
    lengths = list(loader)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"items": len(dataset), "crop_frames": crop_frames, "lengths": lengths}),
        encoding="utf-8",
    )
    return lengths


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
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch-factor must be positive.")
    if args.musical_codebooks <= 0 or args.musical_codebook_dim <= 0:
        raise ValueError("musical-codebooks and musical-codebook-dim must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")
    if args.beat_pos_weight <= 0 or args.downbeat_pos_weight <= 0:
        raise ValueError("beat and downbeat positive weights must be positive")
    if args.max_note_pos_weight < 1:
        raise ValueError("max-note-pos-weight must be at least 1")
    if args.balanced_softmax_tau < 0:
        raise ValueError("balanced-softmax-tau must be non-negative")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("val-ratio must be in [0, 1).")
    torch.manual_seed(args.seed)
    config = TrainingConfig(
        gradient_checkpointing=args.gradient_checkpointing,
        musical_codebooks=args.musical_codebooks,
        musical_codebook_dim=args.musical_codebook_dim,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _build_dataset(args, config)
    train_dataset, val_dataset = _split_dataset(dataset, args.val_ratio, args.seed)
    train_indices = list(train_dataset.indices) if isinstance(train_dataset, Subset) else list(range(len(dataset)))
    loss_stats_path = args.loss_stats_path or args.output_dir / "symbolic_loss_stats.pt"
    loss_statistics = _load_or_compute_symbolic_loss_statistics(
        dataset,
        train_indices,
        config,
        seed=args.seed + 2_000_003,
        path=loss_stats_path,
        recompute=args.recompute_loss_stats,
    )
    binary_pos_weights, categorical_class_counts = _build_symbolic_loss_weights(
        loss_statistics,
        beat_pos_weight=args.beat_pos_weight,
        downbeat_pos_weight=args.downbeat_pos_weight,
        max_note_pos_weight=args.max_note_pos_weight,
    )
    loader_options = {
        "num_workers": args.num_workers,
        "collate_fn": collate_symbolic_sequences,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)

    train_sampler = None
    if args.length_buckets:
        lengths_path = args.output_dir / "token_lengths.json"
        lengths = _sequence_lengths(
            train_dataset,
            args.num_workers,
            lengths_path,
            args.crop_frames if args.crop_frames > 0 else None,
        )
        train_sampler = LengthBucketSampler(lengths, args.batch_size, seed=args.seed)
        print(f"length buckets: {len(train_sampler)} batches from {len(lengths)} items", flush=True)
        loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_options)
    else:
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_options)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_options)
    teacher = SymbolicTeacher(config).to(device)
    if args.compile_encoder:
        # Compile only the event Transformer while preserving checkpoint keys.
        teacher.encoder.encoder.forward = torch.compile(
            teacher.encoder.encoder.forward,
            mode=args.compile_mode,
        )
    criterion = SymbolicTeacherLoss(
        config.symbolic_reconstruction_weight,
        binary_pos_weights=binary_pos_weights,
        categorical_class_counts=categorical_class_counts,
        balanced_softmax_tau=args.balanced_softmax_tau,
    )
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.lr)
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

    source = args.midi_dir if args.midi_dir is not None else args.token_dir
    crop = f"{args.crop_frames} frames" if args.crop_frames > 0 else "whole song"
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requires the optional 'wandb' package. Install it with: pip install wandb"
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                **asdict(config),
                "source": str(source),
                "files": len(dataset),
                "train_files": len(train_dataset),
                "val_files": len(val_dataset) if val_dataset is not None else 0,
                "val_ratio": args.val_ratio,
                "crop_frames": args.crop_frames,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "grad_clip": args.grad_clip,
                "beat_pos_weight": args.beat_pos_weight,
                "downbeat_pos_weight": args.downbeat_pos_weight,
                "max_note_pos_weight": args.max_note_pos_weight,
                "note_onset_pos_weight_mean": float(binary_pos_weights["note_onset"].mean().item()),
                "note_offset_pos_weight_mean": float(binary_pos_weights["note_offset"].mean().item()),
                "balanced_softmax_tau": args.balanced_softmax_tau,
                "loss_stats_path": str(loss_stats_path),
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "length_buckets": args.length_buckets,
                "musical_codebook_dim": config.musical_codebook_dim,
                "device": str(device),
                "amp": use_amp,
                "amp_dtype": str(amp_dtype),
                "seed": args.seed,
                "compile_encoder": args.compile_encoder,
                "compile_mode": args.compile_mode,
            },
        )

    print(
        f"source={source} files={len(dataset)} train={len(train_dataset)} "
        f"val={len(val_dataset) if val_dataset is not None else 0} crop={crop} device={device} "
        f"batch_size={args.batch_size} amp={use_amp} amp_dtype={amp_dtype} wandb={args.wandb} "
        f"onset_pos_weight={binary_pos_weights['note_onset'].mean().item():.2f} "
        f"offset_pos_weight={binary_pos_weights['note_offset'].mean().item():.2f} "
        f"beat_pos_weight={args.beat_pos_weight:.2f} downbeat_pos_weight={args.downbeat_pos_weight:.2f} "
        f"max_note_pos_weight={args.max_note_pos_weight:.2f} "
        f"balanced_softmax_tau={args.balanced_softmax_tau:.2f}"
    )
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                # Re-draw the buckets so batches differ between epochs.
                train_sampler.set_epoch(epoch)
            teacher.train()
            totals: dict[str, float] = {}
            for step, batch in enumerate(loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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
                scaler.scale(loss_values["loss_total"]).backward()
                # Clip true gradients after removing the loss scale.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                batch_metrics = {name: value.detach().item() for name, value in loss_values.items()}
                for name, value in batch_metrics.items():
                    totals[name] = totals.get(name, 0.0) + value

                if step % args.log_interval == 0 or step == len(loader):
                    print(f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} loss={batch_metrics['loss_total']:.4f}")
                    if wandb_run is not None:
                        step_metrics = {
                            f"train/{_loss_metric_name(name)}": value for name, value in batch_metrics.items()
                        }
                        step_metrics.update(
                            {
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                            }
                        )
                        wandb.log(step_metrics, step=global_step)

            denominator = max(1, len(loader))
            average_metrics = {
                f"epoch/{_loss_metric_name(name)}": value / denominator for name, value in totals.items()
            }
            average = average_metrics["epoch/loss"]
            _save_checkpoint(teacher, config, args.output_dir, epoch, average)
            print(f"saved={args.output_dir / 'last.pt'} average_loss={average:.4f}")
            if wandb_run is not None:
                wandb.log(average_metrics, step=global_step)
            if val_loader is not None:
                validation_metrics = _evaluate(
                    teacher,
                    val_loader,
                    criterion,
                    device,
                    use_amp,
                    amp_dtype,
                    validation_seed=args.seed + 1_000_003,
                    codebook_size=config.musical_vocab_size,
                )
                print(
                    "validation "
                    + " ".join(f"{key.removeprefix('val/')}={value:.4f}" for key, value in validation_metrics.items())
                )
                if wandb_run is not None:
                    wandb.log({**validation_metrics, "epoch": epoch}, step=global_step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    train(build_parser().parse_args())
