"""Train the Mel-RVQ target tokenizer on a folder of audio files.

Run from the repository root, for example:

    python -m train.mel_rvq.train --audio-dir data/audio --epochs 10
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from train.mel_rvq.config import MelRVQConfig
from train.mel_rvq.model import MelRVQTokenizer

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}


class AudioClipDataset(Dataset[Tensor]):
    """Load random fixed-length stereo clips from an audio directory."""

    def __init__(
        self,
        audio_dir: str | Path,
        sample_rate: int,
        clip_seconds: float,
        clips_per_epoch: Optional[int] = None,
    ) -> None:
        self.paths = sorted(
            path for path in Path(audio_dir).rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No audio files found under {audio_dir}.")
        if clip_seconds <= 0:
            raise ValueError("clip_seconds must be positive.")

        self.sample_rate = sample_rate
        self.clip_samples = round(sample_rate * clip_seconds)
        self.length = len(self.paths) if clips_per_epoch is None else clips_per_epoch
        if self.length <= 0:
            raise ValueError("clips_per_epoch must be positive.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index % len(self.paths)]
        # Read the dataset directly with libsndfile. This keeps WAV/FLAC/OGG
        # training independent of TorchCodec's FFmpeg shared-library setup.
        samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())

        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=source_rate,
                new_freq=self.sample_rate,
            )

        # Keep the model's stereo contract. Mono files are duplicated; files
        # with more than two channels use their first two channels.
        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.size(0) >= 2:
            waveform = waveform[:2]
        else:
            raise ValueError(f"Audio file has no channels: {path}")

        num_samples = waveform.size(-1)
        if num_samples >= self.clip_samples:
            start = random.randint(0, num_samples - self.clip_samples)
            waveform = waveform[:, start : start + self.clip_samples]
        else:
            waveform = F.pad(waveform, (0, self.clip_samples - num_samples))

        return waveform


def make_config(args: argparse.Namespace) -> MelRVQConfig:
    return MelRVQConfig(
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        temporal_fold=args.temporal_fold,
        acoustic_codebooks=args.codebooks,
        acoustic_vocab_size=args.codebook_size,
        acoustic_rvq_dim=args.rvq_dim,
        rvq_commitment_weight=args.commitment_weight,
        rvq_reconstruction_weight=args.reconstruction_weight,
    )


def mel_frontend_signature(config: MelRVQConfig) -> dict[str, int]:
    """Return the frontend settings that determine Mel statistics."""

    return {
        "sample_rate": config.sample_rate,
        "audio_channels": config.audio_channels,
        "n_mels": config.n_mels,
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
        "temporal_fold": config.temporal_fold,
    }


@torch.no_grad()
def estimate_mel_stats(
    tokenizer: MelRVQTokenizer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Estimate global mean/std before the tokenizer frontend is normalized."""

    tokenizer.eval()
    count = 0
    mean = torch.zeros((), dtype=torch.float64, device=device)
    sum_squared_deviation = torch.zeros((), dtype=torch.float64, device=device)

    for audio in loader:
        mel = tokenizer.frontend(audio.to(device, non_blocking=True)).to(dtype=torch.float64)
        values = mel.reshape(-1)
        batch_count = values.numel()
        if batch_count == 0:
            continue

        batch_mean = values.mean()
        batch_sum_squared_deviation = (values - batch_mean).square().sum()
        if count == 0:
            mean = batch_mean
            sum_squared_deviation = batch_sum_squared_deviation
            count = batch_count
            continue

        total_count = count + batch_count
        delta = batch_mean - mean
        sum_squared_deviation = (
            sum_squared_deviation + batch_sum_squared_deviation + delta.square() * (count * batch_count / total_count)
        )
        mean = mean + delta * (batch_count / total_count)
        count = total_count

    if count == 0:
        raise RuntimeError("Could not estimate Mel statistics from an empty loader.")

    std = torch.sqrt(sum_squared_deviation / count).item()
    return mean.item(), max(std, 1e-6)


def load_or_estimate_mel_stats(
    stats_path: Path,
    tokenizer: MelRVQTokenizer,
    stats_loader: DataLoader | None,
    device: torch.device,
    config: MelRVQConfig,
) -> tuple[float, float, str]:
    """Load cached frontend statistics or estimate and cache them once."""

    signature = mel_frontend_signature(config)
    if stats_path.exists():
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if payload.get("frontend") != signature:
            raise ValueError(
                f"Mel statistics at {stats_path} were made for a different frontend. "
                "Delete the file or pass a matching --stats-path."
            )
        mean = float(payload["mean"])
        std = float(payload["std"])
        return mean, std, "cache"

    if stats_loader is None:
        raise RuntimeError("A statistics loader is required when the cache does not exist.")
    mean, std = estimate_mel_stats(tokenizer, stats_loader, device)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "frontend": signature,
                "mean": mean,
                "std": std,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return mean, std, "estimated"


def save_checkpoint(
    tokenizer: MelRVQTokenizer,
    config: MelRVQConfig,
    output_dir: Path,
    epoch: int,
    loss: float,
) -> None:
    checkpoint = {
        "state_dict": tokenizer.state_dict(),
        "config": asdict(config),
        "mel_stats": {
            "mean": tokenizer.mel_stats[0],
            "std": tokenizer.mel_stats[1],
        },
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, output_dir / "last.pt")
    torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = make_config(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = AudioClipDataset(
        audio_dir=args.audio_dir,
        sample_rate=config.sample_rate,
        clip_seconds=args.clip_seconds,
        clips_per_epoch=args.clips_per_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    tokenizer = MelRVQTokenizer(config).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_path) if args.stats_path is not None else output_dir / "mel_stats.json"
    stats_loader = None
    if not stats_path.exists():
        stats_dataset = AudioClipDataset(
            audio_dir=args.audio_dir,
            sample_rate=config.sample_rate,
            clip_seconds=args.clip_seconds,
            clips_per_epoch=args.stats_clips,
        )
        stats_loader = DataLoader(
            stats_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    mel_mean, mel_std, stats_source = load_or_estimate_mel_stats(
        stats_path,
        tokenizer,
        stats_loader,
        device,
        config,
    )
    tokenizer.set_mel_stats(mel_mean, mel_std)
    print(f"mel_stats mean={mel_mean:.6f} std={mel_std:.6f} source={stats_source} path={stats_path}")

    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=args.lr)

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
                "audio_dir": str(args.audio_dir),
                "clips_per_epoch": len(dataset),
                "batch_size": args.batch_size,
                "clip_seconds": args.clip_seconds,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "seed": args.seed,
                "mel_mean": mel_mean,
                "mel_std": mel_std,
            },
        )

    print(
        f"files={len(dataset.paths)} clips/epoch={len(dataset)} "
        f"device={device} codebooks={config.acoustic_codebooks} "
        f"codebook_size={config.acoustic_vocab_size} "
        f"wandb={args.wandb}"
    )

    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            tokenizer.train()
            total_loss = 0.0
            total_codebook_loss = 0.0
            total_commitment_loss = 0.0
            total_reconstruction_loss = 0.0

            for step, audio in enumerate(loader, start=1):
                audio = audio.to(device, non_blocking=True)
                result = tokenizer(audio)

                optimizer.zero_grad(set_to_none=True)
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), args.grad_clip)
                optimizer.step()

                global_step += 1
                loss = result.loss.detach().item()
                codebook_loss = result.codebook_loss.detach().item()
                commitment_loss = result.commitment_loss.detach().item()
                reconstruction_loss = (
                    result.reconstruction_loss.detach().item() if result.reconstruction_loss is not None else 0.0
                )
                total_loss += loss
                total_codebook_loss += codebook_loss
                total_commitment_loss += commitment_loss
                total_reconstruction_loss += reconstruction_loss

                if step % args.log_interval == 0 or step == len(loader):
                    print(f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} loss={loss:.4f}")
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "train/loss": loss,
                                "train/codebook_loss": codebook_loss,
                                "train/commitment_loss": commitment_loss,
                                "train/reconstruction_loss": reconstruction_loss,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

            denominator = max(len(loader), 1)
            average_loss = total_loss / denominator
            epoch_metrics = {
                "epoch/loss": average_loss,
                "epoch/codebook_loss": total_codebook_loss / denominator,
                "epoch/commitment_loss": total_commitment_loss / denominator,
                "epoch/reconstruction_loss": total_reconstruction_loss / denominator,
            }
            save_checkpoint(tokenizer, config, output_dir, epoch, average_loss)
            print(f"saved={output_dir / 'last.pt'} average_loss={average_loss:.4f}")
            if wandb_run is not None:
                wandb.log(epoch_metrics, step=global_step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files.")
    parser.add_argument("--output-dir", default="checkpoints/mel_rvq")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Mel statistics JSON path. Defaults to <output-dir>/mel_stats.json.",
    )
    parser.add_argument(
        "--stats-clips",
        type=int,
        default=None,
        help="Number of random clips used for statistics; defaults to the dataset length.",
    )
    parser.add_argument(
        "--clips-per-epoch",
        type=int,
        default=None,
        help="Override the number of random clips sampled per epoch.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="For example: cuda or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="tsumugi-mrl-mel-rvq")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)

    # These defaults match MelRVQConfig and the 22,050 Hz / 25 Hz frontend.
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=441)
    parser.add_argument("--temporal-fold", type=int, default=2)
    parser.add_argument("--codebooks", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--rvq-dim", type=int, default=128)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
