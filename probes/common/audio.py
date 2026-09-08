"""Small audio I/O helpers used by the probe scripts.

The model expects stereo 22.05 kHz tensors. Most probe datasets contain
mono files and use a mixture of WAV, OGG, and MP3, so conversion is kept in
one place rather than duplicated in each dataset adapter.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def _ffmpeg_path() -> str:
    bundled = Path(__file__).parents[2] / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    return str(bundled) if bundled.exists() else (shutil.which("ffmpeg") or "ffmpeg")


def _read_with_ffmpeg(
    path: Path,
    sample_rate: int,
    offset_seconds: float,
    duration_seconds: float | None,
) -> torch.Tensor:
    command = [_ffmpeg_path(), "-v", "error"]
    if offset_seconds > 0:
        command += ["-ss", str(offset_seconds)]
    command += ["-i", str(path)]
    if duration_seconds is not None:
        command += ["-t", str(duration_seconds)]
    command += ["-vn", "-ac", "2", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32).clone()
    return audio.reshape(-1, 2).transpose(0, 1).contiguous()


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = 22_050,
    offset_seconds: float = 0.0,
    duration_seconds: float | None = None,
    pad_to_duration: bool = True,
) -> torch.Tensor:
    """Load an audio segment as ``[2, samples]`` float32 at ``sample_rate``."""

    path = Path(path)
    if path.suffix.lower() == ".mp3":
        audio = _read_with_ffmpeg(path, sample_rate, offset_seconds, duration_seconds)
    else:
        info = sf.info(str(path))
        start = max(0, int(round(offset_seconds * info.samplerate)))
        frames = -1 if duration_seconds is None else max(0, int(round(duration_seconds * info.samplerate)))
        data, source_rate = sf.read(str(path), start=start, frames=frames, always_2d=True, dtype="float32")
        audio = torch.from_numpy(data.T.copy())
        if source_rate != sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
        if audio.size(0) == 1:
            audio = audio.repeat(2, 1)
        elif audio.size(0) >= 2:
            audio = audio[:2]
        else:
            audio = torch.zeros(2, 0, dtype=torch.float32)

    if duration_seconds is not None and pad_to_duration:
        target_samples = int(round(duration_seconds * sample_rate))
        if audio.size(1) < target_samples:
            audio = torch.nn.functional.pad(audio, (0, target_samples - audio.size(1)))
        else:
            audio = audio[:, :target_samples]
    return audio.to(dtype=torch.float32).contiguous()
