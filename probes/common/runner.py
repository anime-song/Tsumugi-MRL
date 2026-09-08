"""Feature extraction and command-line plumbing shared by dataset probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from .audio import load_audio
from .linear import train_linear_probe


@dataclass(frozen=True)
class ClipExample:
    path: Path
    label: int | list[float] | float
    split: str
    key: str
    offset_seconds: float = 0.0


@dataclass(frozen=True)
class FrameExample:
    path: Path
    split: str
    key: str
    label_fn: Callable[[torch.Tensor], torch.Tensor]
    duration_seconds: float | None = None
    offset_seconds: float = 0.0


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Exported audio model. With --random-init, only its config and Mel statistics are used.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Construct a randomly initialized frozen encoder instead of loading weights.",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional disk cache for extracted frozen-encoder features.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record cache, extraction, and linear-head timings in the result JSON.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable feature-extraction progress output.",
    )
    parser.add_argument(
        "--audio-workers",
        type=int,
        default=4,
        help="Concurrent audio readers used during clip feature extraction.",
    )
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=8192,
        help="Batch size for the trainable linear head.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for encoder initialization and sampling.")


def _example_signature(examples: list) -> dict[str, object]:
    digest = hashlib.sha256()
    for example in examples:
        duration = (
            ""
            if not isinstance(example, FrameExample) or example.duration_seconds is None
            else f"{float(example.duration_seconds):.9g}"
        )
        record = "\x1f".join(
            (
                type(example).__name__,
                example.split,
                example.key,
                str(example.path),
                f"{float(example.offset_seconds):.9g}",
                duration,
            )
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return {"count": len(examples), "sha256": digest.hexdigest()}


def _cache_metadata(
    *,
    kind: str,
    examples: list,
    checkpoint: str | Path | None,
    random_init: bool,
    seed: int,
    sample_rate: int,
    duration_seconds: float | None = None,
    frame_rate: float | None = None,
    task: str | None = None,
) -> dict[str, object]:
    return {
        "version": 3 if kind == "frame" else 2,
        "kind": kind,
        "task": task,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "initialization": "random" if random_init else "pretrained",
        "seed": seed,
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "frame_rate": frame_rate,
        "examples": _example_signature(examples),
    }


def _load_feature_cache(
    path: str | Path | None,
    metadata: dict[str, object],
    *,
    require_targets: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    stored_metadata = payload.get("metadata")
    if stored_metadata != metadata and isinstance(stored_metadata, dict):
        # Read caches written before the compile option was removed. New
        # caches do not contain this legacy marker.
        legacy_metadata = dict(stored_metadata)
        legacy_metadata.pop("compiled_transformer", None)
        stored_metadata = legacy_metadata
    if stored_metadata != metadata:
        raise ValueError(f"Feature cache metadata mismatch: {path}. Choose a new cache path.")
    features = payload.get("features")
    targets = payload.get("targets")
    if not isinstance(features, dict):
        raise ValueError(f"Feature cache has no feature tensors: {path}")
    if require_targets and not isinstance(targets, dict):
        raise ValueError(f"Frame feature cache has no target tensors: {path}")
    return features, targets if isinstance(targets, dict) else None


def _save_feature_cache(
    path: str | Path | None,
    metadata: dict[str, object],
    features: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor] | None = None,
) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    payload = {"metadata": metadata, "features": features}
    if targets is not None:
        payload["targets"] = targets
    torch.save(payload, temporary)
    temporary.replace(path)


def _clip_targets(examples: list[ClipExample]) -> dict[str, torch.Tensor]:
    targets: dict[str, list[torch.Tensor]] = {}
    for example in examples:
        if isinstance(example.label, list):
            target = torch.tensor(example.label, dtype=torch.float32)
        elif isinstance(example.label, float):
            target = torch.tensor(example.label, dtype=torch.float32)
        else:
            target = torch.tensor(example.label, dtype=torch.long)
        targets.setdefault(example.split, []).append(target)
    return {split: torch.stack(values) for split, values in targets.items()}


def load_audio_model(
    checkpoint: str | Path | None,
    device: str,
    *,
    random_init: bool = False,
) -> torch.nn.Module:
    from tsumugi_mrl import TsumugiMRLModel

    if random_init:
        if checkpoint is None:
            model = TsumugiMRLModel()
        else:
            # Reuse only architecture and frontend normalization; pretrained
            # weights are discarded before feature extraction.
            source = TsumugiMRLModel.from_pretrained(str(checkpoint)).cpu()
            model = TsumugiMRLModel(source.config)
            model.set_mel_stats(*source.audio_encoder.frontend.mel_stats)
            del source
    else:
        if checkpoint is None:
            raise ValueError("Provide --checkpoint or use --random-init.")
        model = TsumugiMRLModel.from_pretrained(str(checkpoint))
    return model.to(device).eval().requires_grad_(False)


def _select_examples(examples: list, max_items: int, seed: int) -> list:
    if max_items <= 0:
        return examples
    rng = random.Random(seed)
    selected: list = []
    for split in sorted({example.split for example in examples}):
        subset = [example for example in examples if example.split == split]
        rng.shuffle(subset)
        selected.extend(subset[:max_items])
    return selected


def _device_name(device: str) -> str:
    return "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _report_progress(label: str, done: int, total: int, started: float, *, final: bool = False) -> None:
    elapsed = time.perf_counter() - started
    rate = done / max(elapsed, 1e-6)
    remaining = (total - done) / max(rate, 1e-6)
    eta = "done" if final else f"ETA {remaining / 60:.1f} min"
    print(
        f"[{label}] {done}/{total} ({100.0 * done / max(total, 1):5.1f}%) elapsed {elapsed / 60:.1f} min, {eta}",
        flush=True,
    )


@torch.inference_mode()
def extract_clip_features(
    model: torch.nn.Module,
    examples: list[ClipExample],
    *,
    sample_rate: int,
    duration_seconds: float,
    batch_size: int,
    device: str,
    audio_workers: int = 1,
    progress: bool = True,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    features: dict[str, list[torch.Tensor]] = {}
    targets: dict[str, list[torch.Tensor]] = {}
    model_device = torch.device(_device_name(device))
    started = time.perf_counter()
    report_every = max(1, len(examples) // 20)
    last_report = started
    processed = 0

    def read_audio(example: ClipExample) -> torch.Tensor:
        return load_audio(
            example.path,
            sample_rate=sample_rate,
            offset_seconds=example.offset_seconds,
            duration_seconds=duration_seconds,
        )

    executor = ThreadPoolExecutor(max_workers=max(1, audio_workers)) if audio_workers > 1 else None
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            load_started = time.perf_counter()
            audio_values = (
                list(executor.map(read_audio, batch))
                if executor is not None
                else [read_audio(example) for example in batch]
            )
            audio_cpu = torch.stack(audio_values)
            if timings is not None:
                timings["audio_loading_seconds"] += time.perf_counter() - load_started
            transfer_started = time.perf_counter()
            audio = audio_cpu.to(model_device)
            _synchronize(str(model_device))
            if timings is not None:
                timings["device_transfer_seconds"] += time.perf_counter() - transfer_started
            encode_started = time.perf_counter()
            hidden = model.encode_audio(audio)
            _synchronize(str(model_device))
            if timings is not None:
                timings["encoder_seconds"] += time.perf_counter() - encode_started
            pooled = model.mean_pool(hidden).cpu()
            for example, feature in zip(batch, pooled, strict=True):
                features.setdefault(example.split, []).append(feature)
                if isinstance(example.label, list):
                    target = torch.tensor(example.label, dtype=torch.float32)
                elif isinstance(example.label, float):
                    target = torch.tensor(example.label, dtype=torch.float32)
                else:
                    target = torch.tensor(example.label, dtype=torch.long)
                targets.setdefault(example.split, []).append(target)
            processed += len(batch)
            now = time.perf_counter()
            if progress and (processed % report_every == 0 or processed == len(examples) or now - last_report >= 15.0):
                _report_progress("clip features", processed, len(examples), started, final=processed == len(examples))
                last_report = now
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return (
        {split: torch.stack(values) for split, values in features.items()},
        {split: torch.stack(values) for split, values in targets.items()},
    )


@torch.inference_mode()
def extract_frame_features(
    model: torch.nn.Module,
    examples: list[FrameExample],
    *,
    sample_rate: int,
    frame_rate: float,
    batch_size: int,
    device: str,
    audio_workers: int = 1,
    progress: bool = True,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    features: dict[str, list[torch.Tensor]] = {}
    targets: dict[str, list[torch.Tensor]] = {}
    model_device = torch.device(_device_name(device))
    config = getattr(model, "config", None)
    n_fft = int(getattr(config, "n_fft", 2048))
    hop_length = int(getattr(config, "hop_length", 441))
    temporal_fold = int(getattr(config, "temporal_fold", 2))

    def token_count(duration_seconds: float) -> int:
        samples = int(round(duration_seconds * sample_rate))
        mel_frames = max(0, (samples - n_fft) // hop_length + 1)
        return (mel_frames - mel_frames % temporal_fold) // temporal_fold

    started = time.perf_counter()
    report_every = max(1, len(examples) // 20)
    last_report = started
    processed = 0
    executor = ThreadPoolExecutor(max_workers=max(1, audio_workers)) if audio_workers > 1 else None

    def read_audio(example: FrameExample, duration: float | None) -> torch.Tensor:
        return load_audio(
            example.path,
            sample_rate=sample_rate,
            offset_seconds=example.offset_seconds,
            duration_seconds=duration,
            pad_to_duration=duration is not None,
        )

    try:
        for start in range(0, len(examples), max(1, batch_size)):
            batch = examples[start : start + max(1, batch_size)]
            # Padding lets the GPU process several recordings together. The
            # padding mask prevents zero-padded tails from affecting valid
            # frames, which are sliced back to each recording's true length.
            if any(example.duration_seconds is None for example in batch):
                batch = batch[:1]
                common_duration = None
            else:
                common_duration = max(float(example.duration_seconds) for example in batch)
            load_started = time.perf_counter()
            audio_values = (
                list(executor.map(lambda example: read_audio(example, common_duration), batch))
                if executor is not None
                else [read_audio(example, common_duration) for example in batch]
            )
            audio_cpu = torch.stack(audio_values)
            if timings is not None:
                timings["audio_loading_seconds"] += time.perf_counter() - load_started
            valid_lengths = torch.tensor(
                [
                    token_count(float(example.duration_seconds))
                    if example.duration_seconds is not None
                    else audio_cpu.size(-1) // hop_length // temporal_fold
                    for example in batch
                ],
                dtype=torch.long,
            )
            max_length = token_count(common_duration) if common_duration is not None else valid_lengths.max().item()
            padding_mask = torch.arange(max_length).unsqueeze(0) >= valid_lengths.unsqueeze(1)
            transfer_started = time.perf_counter()
            audio = audio_cpu.to(model_device)
            padding_mask = padding_mask.to(model_device) if padding_mask.any() else None
            _synchronize(str(model_device))
            if timings is not None:
                timings["device_transfer_seconds"] += time.perf_counter() - transfer_started
            encode_started = time.perf_counter()
            hidden = model.encode_audio(audio, padding_mask=padding_mask)
            _synchronize(str(model_device))
            hidden = hidden.cpu()
            if timings is not None:
                timings["encoder_seconds"] += time.perf_counter() - encode_started
            for index, example in enumerate(batch):
                count = min(int(valid_lengths[index].item()), hidden.size(1))
                frame_times = (torch.arange(count, dtype=torch.float32) + 0.5) / frame_rate
                labels = example.label_fn(frame_times + example.offset_seconds)
                count = min(count, labels.size(0))
                if count > 0:
                    features.setdefault(example.split, []).append(hidden[index, :count])
                    targets.setdefault(example.split, []).append(labels[:count])
                processed += 1
                now = time.perf_counter()
                if progress and (
                    processed % report_every == 0 or processed == len(examples) or now - last_report >= 15.0
                ):
                    _report_progress(
                        "frame features", processed, len(examples), started, final=processed == len(examples)
                    )
                    last_report = now
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return (
        {split: torch.cat(values) for split, values in features.items()},
        {split: torch.cat(values) for split, values in targets.items()},
    )


def run_clip_probe(
    examples: list[ClipExample],
    *,
    checkpoint: str | Path | None,
    task: str,
    output_dim: int,
    duration_seconds: float,
    sample_rate: int = 22_050,
    batch_size: int = 4,
    epochs: int = 30,
    learning_rate: float = 1e-2,
    max_items: int = 0,
    seed: int = 0,
    device: str = "auto",
    output: str | Path | None = None,
    random_init: bool = False,
    feature_cache: str | Path | None = None,
    profile: bool = False,
    progress: bool = True,
    audio_workers: int = 1,
    head_batch_size: int = 8192,
) -> dict:
    total_started = time.perf_counter()
    torch.manual_seed(seed)
    examples = _select_examples(examples, max_items, seed)
    model_device = _device_name(device)
    timings = (
        {
            "cache_load_seconds": 0.0,
            "model_setup_seconds": 0.0,
            "audio_loading_seconds": 0.0,
            "device_transfer_seconds": 0.0,
            "encoder_seconds": 0.0,
            "feature_extraction_seconds": 0.0,
            "cache_save_seconds": 0.0,
            "linear_probe_seconds": 0.0,
        }
        if profile
        else None
    )
    cache_metadata = _cache_metadata(
        kind="clip",
        examples=examples,
        checkpoint=checkpoint,
        random_init=random_init,
        seed=seed,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )
    cache_started = time.perf_counter()
    cached = _load_feature_cache(feature_cache, cache_metadata, require_targets=False)
    if timings is not None:
        timings["cache_load_seconds"] = time.perf_counter() - cache_started
    if cached is None:
        if progress and feature_cache is not None:
            print(f"[feature cache] miss: {feature_cache}", flush=True)
        model_started = time.perf_counter()
        model = load_audio_model(
            checkpoint,
            model_device,
            random_init=random_init,
        )
        if timings is not None:
            timings["model_setup_seconds"] = time.perf_counter() - model_started
        extraction_started = time.perf_counter()
        features, _ = extract_clip_features(
            model,
            examples,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            batch_size=batch_size,
            device=model_device,
            audio_workers=audio_workers,
            progress=progress,
            timings=timings,
        )
        if timings is not None:
            timings["feature_extraction_seconds"] = time.perf_counter() - extraction_started
        cache_save_started = time.perf_counter()
        _save_feature_cache(feature_cache, cache_metadata, features)
        if timings is not None:
            timings["cache_save_seconds"] = time.perf_counter() - cache_save_started
    else:
        if progress:
            print(f"[feature cache] hit: {feature_cache}", flush=True)
        features, _ = cached
    targets = _clip_targets(examples)
    if "train" not in features:
        raise ValueError("A train split is required for a linear probe.")
    eval_sets = {split: (features[split], targets[split]) for split in sorted(features) if split != "train"}
    if progress:
        print(
            f"[linear probe] start: {features['train'].shape[0]} train examples, {epochs} epochs",
            flush=True,
        )
    training_info: dict[str, object] = {}
    linear_started = time.perf_counter()
    results = train_linear_probe(
        features["train"],
        targets["train"],
        eval_sets,
        task=task,
        output_dim=output_dim,
        epochs=epochs,
        batch_size=head_batch_size,
        learning_rate=learning_rate,
        device=model_device,
        patience=10,
        validation_split=0.1,
        seed=seed,
        training_info=training_info,
    )
    _synchronize(model_device)
    if timings is not None:
        timings["linear_probe_seconds"] = time.perf_counter() - linear_started
    if progress:
        print(
            f"[linear probe] done: {training_info['epochs_run']}/{epochs} epochs, "
            f"best={training_info['best_epoch']}, {(time.perf_counter() - linear_started) / 60:.1f} min",
            flush=True,
        )
    payload = {
        "task": task,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "initialization": "random" if random_init else "pretrained",
        "feature_cache": str(feature_cache) if feature_cache is not None else None,
        "epochs": epochs,
        "head_batch_size": head_batch_size,
        "audio_workers": audio_workers,
        "seed": seed,
        "num_examples": {split: int(value.size(0)) for split, value in features.items()},
        "metrics": results,
        "training": training_info,
    }
    if timings is not None:
        timings["total_seconds"] = time.perf_counter() - total_started
        payload["timings_seconds"] = timings
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_frame_probe(
    examples: list[FrameExample],
    *,
    checkpoint: str | Path | None,
    task: str,
    output_dim: int,
    sample_rate: int = 22_050,
    frame_rate: float = 25.0,
    batch_size: int = 1,
    epochs: int = 30,
    learning_rate: float = 1e-2,
    max_items: int = 0,
    seed: int = 0,
    device: str = "auto",
    output: str | Path | None = None,
    random_init: bool = False,
    feature_cache: str | Path | None = None,
    profile: bool = False,
    progress: bool = True,
    audio_workers: int = 1,
    head_batch_size: int = 8192,
) -> dict:
    total_started = time.perf_counter()
    torch.manual_seed(seed)
    examples = _select_examples(examples, max_items, seed)
    model_device = _device_name(device)
    timings = (
        {
            "cache_load_seconds": 0.0,
            "model_setup_seconds": 0.0,
            "audio_loading_seconds": 0.0,
            "device_transfer_seconds": 0.0,
            "encoder_seconds": 0.0,
            "feature_extraction_seconds": 0.0,
            "cache_save_seconds": 0.0,
            "linear_probe_seconds": 0.0,
        }
        if profile
        else None
    )
    cache_metadata = _cache_metadata(
        kind="frame",
        examples=examples,
        checkpoint=checkpoint,
        random_init=random_init,
        seed=seed,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        task=task,
    )
    cache_started = time.perf_counter()
    cached = _load_feature_cache(feature_cache, cache_metadata, require_targets=True)
    if timings is not None:
        timings["cache_load_seconds"] = time.perf_counter() - cache_started
    if cached is None:
        if progress and feature_cache is not None:
            print(f"[feature cache] miss: {feature_cache}", flush=True)
        model_started = time.perf_counter()
        model = load_audio_model(
            checkpoint,
            model_device,
            random_init=random_init,
        )
        if timings is not None:
            timings["model_setup_seconds"] = time.perf_counter() - model_started
        extraction_started = time.perf_counter()
        features, targets = extract_frame_features(
            model,
            examples,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            batch_size=batch_size,
            device=model_device,
            audio_workers=audio_workers,
            progress=progress,
            timings=timings,
        )
        if timings is not None:
            timings["feature_extraction_seconds"] = time.perf_counter() - extraction_started
        cache_save_started = time.perf_counter()
        _save_feature_cache(feature_cache, cache_metadata, features, targets)
        if timings is not None:
            timings["cache_save_seconds"] = time.perf_counter() - cache_save_started
    else:
        if progress:
            print(f"[feature cache] hit: {feature_cache}", flush=True)
        features, targets = cached
        assert targets is not None
    if progress:
        print(
            f"[linear probe] start: {features['train'].shape[0]} train frames, {epochs} epochs",
            flush=True,
        )
    training_info: dict[str, object] = {}
    linear_started = time.perf_counter()
    eval_sets = {split: (features[split], targets[split]) for split in sorted(features) if split != "train"}
    results = train_linear_probe(
        features["train"],
        targets["train"],
        eval_sets,
        task=task,
        output_dim=output_dim,
        epochs=epochs,
        batch_size=head_batch_size,
        learning_rate=learning_rate,
        device=model_device,
        patience=10,
        validation_split=0.1,
        seed=seed,
        training_info=training_info,
    )
    _synchronize(model_device)
    if timings is not None:
        timings["linear_probe_seconds"] = time.perf_counter() - linear_started
    if progress:
        print(
            f"[linear probe] done: {training_info['epochs_run']}/{epochs} epochs, "
            f"best={training_info['best_epoch']}, {(time.perf_counter() - linear_started) / 60:.1f} min",
            flush=True,
        )
    payload = {
        "task": task,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "initialization": "random" if random_init else "pretrained",
        "feature_cache": str(feature_cache) if feature_cache is not None else None,
        "epochs": epochs,
        "head_batch_size": head_batch_size,
        "audio_workers": audio_workers,
        "seed": seed,
        "num_frames": {split: int(value.size(0)) for split, value in features.items()},
        "metrics": results,
        "training": training_info,
    }
    if timings is not None:
        timings["total_seconds"] = time.perf_counter() - total_started
        payload["timings_seconds"] = timings
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
