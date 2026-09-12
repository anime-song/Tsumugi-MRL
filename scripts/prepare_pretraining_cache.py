"""Build compact on-disk caches for fast audio pretraining."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.loading import MEL_RVQ_SOURCE, load_teacher
from train.mel_rvq.model import MelRVQTokenizer
from train.symbolic import save_pretraining_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/symbolic/manifest.json"))
    parser.add_argument(
        "--mel-checkpoint",
        help=f"Training .pt, export directory, or Hub id. Defaults to {MEL_RVQ_SOURCE}.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/symbolic/pretraining_cache"))
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-frames", type=int, default=750, help="Folded Mel frames processed at once.")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--skip-mel", action="store_true", help="Build only the compact symbolic cache.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def _feature_count(sample_count: int, *, n_fft: int, hop_length: int, temporal_fold: int) -> int:
    mel_frames = (sample_count - n_fft) // hop_length + 1
    return max(0, mel_frames // temporal_fold)


def _write_mel_cache(
    audio_path: Path,
    output_path: Path,
    frontend: torch.nn.Module,
    config,
    device: torch.device,
    dtype: np.dtype,
    chunk_frames: int,
) -> int:
    info = sf.info(audio_path)
    if info.samplerate != config.sample_rate:
        raise ValueError(
            f"{audio_path} has sample rate {info.samplerate}; "
            f"fast feature caching currently requires {config.sample_rate}."
        )
    total_frames = _feature_count(
        info.frames,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        temporal_fold=config.temporal_fold,
    )
    if total_frames <= 0:
        raise ValueError(f"Audio is too short for the Mel frontend: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.npy")
    if temporary.exists():
        temporary.unlink()
    feature_dim = config.temporal_fold * config.audio_channels * config.n_mels
    mel_cache = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=dtype,
        shape=(total_frames, feature_dim),
    )
    samples_per_frame = config.hop_length * config.temporal_fold
    with sf.SoundFile(audio_path) as sound_file:
        for start_frame in range(0, total_frames, chunk_frames):
            frame_count = min(chunk_frames, total_frames - start_frame)
            start_sample = start_frame * samples_per_frame
            sample_count = frame_count * samples_per_frame + config.n_fft - config.hop_length
            sound_file.seek(start_sample)
            samples = sound_file.read(sample_count, dtype="float32", always_2d=True)
            if samples.shape[0] < sample_count:
                raise ValueError(f"Unexpected short read from {audio_path} at frame {start_frame}.")
            audio = torch.from_numpy(samples.T.copy())
            if audio.size(0) == 1:
                audio = audio.repeat(2, 1)
            else:
                audio = audio[:2]
            with torch.inference_mode():
                features = frontend(audio.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            if features.shape != (frame_count, feature_dim):
                raise ValueError(
                    f"Unexpected feature shape for {audio_path}: "
                    f"expected {(frame_count, feature_dim)}, got {features.shape}"
                )
            mel_cache[start_frame : start_frame + frame_count] = features.astype(dtype, copy=False)
    mel_cache.flush()
    del mel_cache
    os.replace(temporary, output_path)
    return total_frames


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    return f"{int(max(0, seconds)) // 60:02d}:{int(max(0, seconds)) % 60:02d}"


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("chunk-frames must be positive.")
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_manifest = (args.output_manifest or output_dir / "manifest.json").resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    successful = [entry for entry in entries if entry.get("status") != "error"]
    if not successful:
        raise ValueError("Manifest contains no successful entries.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = None if args.skip_mel else load_teacher(MelRVQTokenizer, args.mel_checkpoint, MEL_RVQ_SOURCE)
    frontend = tokenizer.frontend.to(device).eval() if tokenizer is not None else None
    dtype = np.float16 if args.dtype == "float16" else np.float32
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tokens").mkdir(parents=True, exist_ok=True)
    if not args.skip_mel:
        (output_dir / "mel").mkdir(parents=True, exist_ok=True)

    output_entries: list[dict[str, object]] = []
    started = time.monotonic()
    for index, entry in enumerate(entries, start=1):
        output_entry = dict(entry)
        if entry.get("status") == "error":
            output_entries.append(output_entry)
            continue
        audio_path = _resolve(manifest, entry["audio_path"])
        token_path = _resolve(manifest, entry["token_path"])
        compact_token_path = output_dir / "tokens" / f"{index - 1:06d}.pt"

        info = sf.info(audio_path)
        if args.overwrite or not compact_token_path.is_file():
            save_pretraining_cache(token_path, compact_token_path)

        output_entry.update(
            {
                "audio_path": _relative(audio_path, output_manifest.parent),
                "audio_sample_rate": int(info.samplerate),
                "audio_frames": int(info.frames),
                "pretraining_token_path": _relative(compact_token_path, output_manifest.parent),
            }
        )
        if not args.skip_mel:
            mel_path = output_dir / "mel" / f"{index - 1:06d}.npy"
            if args.overwrite or not mel_path.is_file():
                mel_frames = _write_mel_cache(
                    audio_path,
                    mel_path,
                    frontend,
                    tokenizer.config,
                    device,
                    dtype,
                    args.chunk_frames,
                )
            else:
                mel_frames = int(np.load(mel_path, mmap_mode="r").shape[0])
            output_entry.update(
                {
                    "mel_path": _relative(mel_path, output_manifest.parent),
                    "mel_frames": mel_frames,
                }
            )
        output_entries.append(output_entry)
        completed = len(output_entries)
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (len(entries) - completed) / rate if rate else None
        print(
            f"cache {completed}/{len(entries)} ({completed / len(entries) * 100:.1f}%) "
            f"elapsed={_format_eta(elapsed)} eta={_format_eta(remaining)}",
            flush=True,
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(output_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved fast pretraining manifest: {output_manifest}", flush=True)


if __name__ == "__main__":
    main()
