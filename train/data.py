"""Shared audio/MIDI windows for masked-audio pretraining."""

from dataclasses import dataclass
import json
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from .config import TrainingConfig
from .symbolic import (
    SymbolicSequence, collate_symbolic_sequences, crop_symbolic_sequence, load_symbolic_cache,
)


@dataclass(frozen=True)
class PretrainingWindow:
    audio: Tensor  # [channels, samples], unmasked audio for both acoustic paths
    symbolic: SymbolicSequence  # MIDI context from the same time interval
    symbolic_frame_indices: Tensor  # [T_audio], local MIDI bins at Mel centers


def crop_pretraining_window(
    audio: Tensor,
    sequence: SymbolicSequence,
    start_frame: int,
    num_frames: int,
    config: TrainingConfig,
) -> PretrainingWindow:
    """Crop a shared interval before running either teacher.

    Audio must already be stereo at config.sample_rate; MIDI must use
    config.symbolic_frame_rate. Trailing incomplete MIDI/audio frames are
    excluded. Use the returned indices to gather symbolic codes onto the
    acoustic prediction axis, rather than truncating two unequal sequences.
    """
    if audio.ndim != 2 or audio.size(0) != config.audio_channels:
        raise ValueError("audio must have shape [channels, samples].")
    if config.symbolic_frame_rate != config.encoder_frame_rate:
        raise ValueError("Paired windows require matching symbolic and encoder frame rates.")
    samples_per_frame = config.hop_length * config.temporal_fold
    available_frames = min(sequence.num_frames, audio.size(-1) // samples_per_frame)
    if start_frame < 0 or start_frame >= available_frames or num_frames <= 0:
        raise ValueError("Requested window must start inside both audio and MIDI and have positive length.")
    num_frames = min(num_frames, available_frames - start_frame)
    start_sample = start_frame * samples_per_frame
    sample_count = num_frames * samples_per_frame
    mel_frames = (sample_count - config.n_fft) // config.hop_length + 1
    audio_frames = mel_frames // config.temporal_fold
    if audio_frames <= 0:
        raise ValueError("Window is too short to form a folded Mel token.")

    # center=False STFT: average the centers of the folded Mel windows.
    # MIDI targets label 40 ms bins, so choose the bin containing that center.
    centers = (
        torch.arange(audio_frames, dtype=torch.float64) * samples_per_frame
        + config.n_fft / 2
        + (config.temporal_fold - 1) * config.hop_length / 2
    )
    symbolic_frame_indices = (centers / samples_per_frame).floor().long()
    return PretrainingWindow(
        audio=audio[:, start_sample : start_sample + sample_count],
        symbolic=crop_symbolic_sequence(sequence, start_frame, num_frames, frame_rate=config.symbolic_frame_rate),
        symbolic_frame_indices=symbolic_frame_indices,
    )


class PairedAudioDataset(Dataset[PretrainingWindow]):
    """Read audio/cache pairs from the preparation manifest and crop jointly."""

    def __init__(self, manifest: str | Path, config: TrainingConfig, crop_frames: int):
        if crop_frames <= 0:
            raise ValueError("crop_frames must be positive.")
        self.config = config
        self.crop_frames = crop_frames
        # Preparation writes absolute paths; custom manifests may use paths
        # relative to the manifest directory.
        manifest = Path(manifest)
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        self.pairs = []
        for entry in entries:
            if entry.get("status") == "error":
                continue
            paths = [Path(entry[key]) for key in ("audio_path", "token_path")]
            paths = [p if p.is_absolute() else manifest.parent / p for p in paths]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            self.pairs.append(tuple(paths))
        if not self.pairs:
            raise ValueError("Manifest contains no successful audio/cache pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> PretrainingWindow:
        audio_path, token_path = self.pairs[index]
        samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(samples.T.copy())
        if rate != self.config.sample_rate:
            audio = torchaudio.functional.resample(audio, rate, self.config.sample_rate)
        audio = audio.repeat(2, 1) if audio.size(0) == 1 else audio[:2]
        sequence = load_symbolic_cache(token_path)
        available = min(sequence.num_frames, audio.size(-1) // (
            self.config.hop_length * self.config.temporal_fold
        ))
        start = int(torch.randint(max(1, available - self.crop_frames + 1), ()).item())
        return crop_pretraining_window(audio, sequence, start, self.crop_frames, self.config)


def collate_pretraining_windows(windows: list[PretrainingWindow]) -> dict[str, Tensor]:
    symbolic = collate_symbolic_sequences([w.symbolic for w in windows])
    batch = {"symbolic_" + key: value for key, value in symbolic.items() if isinstance(value, Tensor)}
    max_samples = max(w.audio.size(-1) for w in windows)
    max_frames = max(w.symbolic_frame_indices.numel() for w in windows)
    batch["audio"] = torch.stack([F.pad(w.audio, (0, max_samples - w.audio.size(-1))) for w in windows])
    # Audio tokens and MIDI FRAME rows have different lengths. Keep a separate
    # audio padding mask and gather map [B, T_audio] for the symbolic codes.
    batch["audio_padding_mask"] = torch.ones(len(windows), max_frames, dtype=torch.bool)
    batch["symbolic_frame_indices"] = torch.zeros(len(windows), max_frames, dtype=torch.long)
    for row, window in enumerate(windows):
        length = window.symbolic_frame_indices.numel()
        batch["audio_padding_mask"][row, :length] = False
        batch["symbolic_frame_indices"][row, :length] = window.symbolic_frame_indices
    return batch
