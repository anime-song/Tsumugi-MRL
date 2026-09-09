"""Shared audio/MIDI windows for masked-audio pretraining."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from .config import TrainingConfig
from .symbolic import (
    SymbolicSequence,
    collate_symbolic_sequences,
    crop_symbolic_sequence,
    load_pretraining_cache,
    load_symbolic_cache,
)


@dataclass(frozen=True)
class PretrainingWindow:
    audio: Tensor | None  # [channels, samples], omitted when Mel features are cached
    symbolic: SymbolicSequence  # MIDI context from the same time interval
    symbolic_frame_indices: Tensor  # [T_audio], local MIDI bins at Mel centers
    mel_features: Tensor | None = None  # [T_audio, D_mel], optional disk-cache input


@dataclass(frozen=True)
class PairedAudioEntry:
    audio_path: Path | None
    token_path: Path
    compact_tokens: bool
    mel_path: Path | None
    audio_sample_rate: int | None
    audio_frames: int | None
    mel_frames: int | None


def _audio_frame_indices(audio_frames: int, config: TrainingConfig) -> Tensor:
    """Map folded Mel frames back to the local symbolic 25 Hz bins."""

    samples_per_frame = config.hop_length * config.temporal_fold
    centers = (
        torch.arange(audio_frames, dtype=torch.float64) * samples_per_frame
        + config.n_fft / 2
        + (config.temporal_fold - 1) * config.hop_length / 2
    )
    return (centers / samples_per_frame).floor().long()


def _audio_frames_for_crop(num_frames: int, config: TrainingConfig) -> int:
    samples_per_frame = config.hop_length * config.temporal_fold
    sample_count = num_frames * samples_per_frame
    mel_frames = (sample_count - config.n_fft) // config.hop_length + 1
    return mel_frames // config.temporal_fold


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
    symbolic_frame_indices = _audio_frame_indices(audio_frames, config)
    return PretrainingWindow(
        audio=audio[:, start_sample : start_sample + sample_count],
        symbolic=crop_symbolic_sequence(sequence, start_frame, num_frames, frame_rate=config.symbolic_frame_rate),
        symbolic_frame_indices=symbolic_frame_indices,
    )


def crop_pretraining_mel_window(
    mel_features: Tensor,
    sequence: SymbolicSequence,
    start_frame: int,
    num_frames: int,
    config: TrainingConfig,
) -> PretrainingWindow:
    """Build a paired window from an already cached folded-Mel slice.

    The cache is generated at the same folded frame rate as the audio
    frontend. The waveform is intentionally omitted so the training loader
    does not decode WAV files or run STFT on every epoch.
    """

    if mel_features.ndim != 2 or mel_features.size(0) <= 0:
        raise ValueError("mel_features must have shape [frames, features].")
    expected_frames = _audio_frames_for_crop(num_frames, config)
    if mel_features.size(0) != expected_frames:
        raise ValueError(f"Expected {expected_frames} cached Mel frames, got {mel_features.size(0)}.")
    return PretrainingWindow(
        audio=None,
        symbolic=crop_symbolic_sequence(sequence, start_frame, num_frames, frame_rate=config.symbolic_frame_rate),
        symbolic_frame_indices=_audio_frame_indices(expected_frames, config),
        mel_features=mel_features,
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
        self.pairs: list[PairedAudioEntry] = []
        for entry in entries:
            if entry.get("status") == "error":
                continue
            token_key = "pretraining_token_path" if entry.get("pretraining_token_path") else "token_path"
            token_path = Path(entry[token_key])
            token_path = token_path if token_path.is_absolute() else manifest.parent / token_path
            if not token_path.is_file():
                raise FileNotFoundError(token_path)
            compact_tokens = token_key == "pretraining_token_path"

            mel_path = entry.get("mel_path")
            if mel_path is not None:
                mel_path = Path(mel_path)
                mel_path = mel_path if mel_path.is_absolute() else manifest.parent / mel_path
                if not mel_path.is_file():
                    raise FileNotFoundError(mel_path)
            audio_path = entry.get("audio_path")
            if audio_path is not None:
                audio_path = Path(audio_path)
                audio_path = audio_path if audio_path.is_absolute() else manifest.parent / audio_path
                if mel_path is None and not audio_path.is_file():
                    raise FileNotFoundError(audio_path)
            if mel_path is None and audio_path is None:
                raise ValueError(f"Manifest entry has neither audio_path nor mel_path: {entry}")

            audio_sample_rate = entry.get("audio_sample_rate")
            audio_frames = entry.get("audio_frames")
            if mel_path is None and (audio_sample_rate is None or audio_frames is None):
                # Read metadata once while constructing the dataset instead of
                # issuing sf.info() for every item in every epoch.
                info = sf.info(audio_path)
                audio_sample_rate = info.samplerate
                audio_frames = info.frames
            self.pairs.append(
                PairedAudioEntry(
                    audio_path=audio_path,
                    token_path=token_path,
                    compact_tokens=compact_tokens,
                    mel_path=mel_path,
                    audio_sample_rate=int(audio_sample_rate) if audio_sample_rate is not None else None,
                    audio_frames=int(audio_frames) if audio_frames is not None else None,
                    mel_frames=int(entry["mel_frames"]) if entry.get("mel_frames") is not None else None,
                )
            )
        if not self.pairs:
            raise ValueError("Manifest contains no successful audio/cache pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> PretrainingWindow:
        entry = self.pairs[index]
        sequence = (
            load_pretraining_cache(entry.token_path) if entry.compact_tokens else load_symbolic_cache(entry.token_path)
        )
        samples_per_frame = self.config.hop_length * self.config.temporal_fold
        if entry.mel_path is not None:
            mel = np.load(entry.mel_path, mmap_mode="r")
            expected_dim = self.config.temporal_fold * self.config.audio_channels * self.config.n_mels
            if mel.ndim != 2 or mel.shape[1] != expected_dim:
                raise ValueError(f"Unexpected Mel cache shape for {entry.mel_path}: {mel.shape}")
            available = min(sequence.num_frames, int(mel.shape[0]))
            crop_frames = min(self.crop_frames, available)
            window_start = int(torch.randint(max(1, available - crop_frames + 1), ()).item())
            audio_frames = _audio_frames_for_crop(crop_frames, self.config)
            mel_window = torch.from_numpy(
                np.array(mel[window_start : window_start + audio_frames], dtype=np.float32, copy=True)
            )
            return crop_pretraining_mel_window(
                mel_window,
                sequence,
                window_start,
                crop_frames,
                self.config,
            )

        audio_path = entry.audio_path
        if audio_path is None or entry.audio_sample_rate is None or entry.audio_frames is None:
            raise RuntimeError("Audio cache entry is missing waveform metadata.")
        if entry.audio_sample_rate == self.config.sample_rate:
            # Read only the random crop instead of decoding the complete song.
            # This is important for the long audio files in the preparation
            # manifest; resampled inputs use the safe full-read fallback below.
            available = min(sequence.num_frames, entry.audio_frames // samples_per_frame)
            window_start = int(torch.randint(max(1, available - self.crop_frames + 1), ()).item())
            start_sample = window_start * samples_per_frame
            sample_count = self.crop_frames * samples_per_frame
            with sf.SoundFile(audio_path) as sound_file:
                sound_file.seek(start_sample)
                samples = sound_file.read(sample_count, dtype="float32", always_2d=True)
            audio = torch.from_numpy(samples.T.copy())
            sequence = crop_symbolic_sequence(
                sequence,
                window_start,
                self.crop_frames,
                frame_rate=self.config.symbolic_frame_rate,
            )
            window_start = 0
        else:
            samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
            audio = torch.from_numpy(samples.T.copy())
            audio = torchaudio.functional.resample(audio, rate, self.config.sample_rate)
            available = min(sequence.num_frames, audio.size(-1) // samples_per_frame)
            window_start = int(torch.randint(max(1, available - self.crop_frames + 1), ()).item())
            # Keep the full resampled waveform; crop_pretraining_window applies
            # window_start to both the waveform and symbolic sequence below.

        audio = audio.repeat(2, 1) if audio.size(0) == 1 else audio[:2]
        return crop_pretraining_window(audio, sequence, window_start, self.crop_frames, self.config)


def collate_pretraining_windows(windows: list[PretrainingWindow]) -> dict[str, Tensor]:
    symbolic = collate_symbolic_sequences([w.symbolic for w in windows])
    batch = {"symbolic_" + key: value for key, value in symbolic.items() if isinstance(value, Tensor)}
    max_frames = max(w.symbolic_frame_indices.numel() for w in windows)
    has_mel_features = windows[0].mel_features is not None
    if any((window.mel_features is not None) != has_mel_features for window in windows):
        raise ValueError("A batch cannot mix waveform and cached-Mel windows.")
    if has_mel_features:
        max_mel_frames = max(window.mel_features.size(0) for window in windows)
        batch["mel_features"] = torch.stack(
            [
                F.pad(
                    window.mel_features,
                    (0, 0, 0, max_mel_frames - window.mel_features.size(0)),
                )
                for window in windows
            ]
        )
    else:
        max_samples = max(window.audio.size(-1) for window in windows)
        batch["audio"] = torch.stack(
            [F.pad(window.audio, (0, max_samples - window.audio.size(-1))) for window in windows]
        )
    # Audio tokens and MIDI FRAME rows have different lengths. Keep a separate
    # audio padding mask and gather map [B, T_audio] for the symbolic codes.
    batch["audio_padding_mask"] = torch.ones(len(windows), max_frames, dtype=torch.bool)
    batch["symbolic_frame_indices"] = torch.zeros(len(windows), max_frames, dtype=torch.long)
    for row, window in enumerate(windows):
        length = window.symbolic_frame_indices.numel()
        batch["audio_padding_mask"][row, :length] = False
        batch["symbolic_frame_indices"][row, :length] = window.symbolic_frame_indices
    return batch
