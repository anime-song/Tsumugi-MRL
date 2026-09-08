"""Run clip-level linear probes on NSynth.

Examples:
    uv run --extra probe python -m probes.nsynth.probe --task pitch --checkpoint checkpoints/pretraining/audio_model
    uv run --extra probe python -m probes.nsynth.probe --task family --checkpoint checkpoints/pretraining/audio_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probes.common.runner import ClipExample, add_model_arguments, run_clip_probe

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "evaluation" / "nsynth"


def _split_directories() -> list[tuple[str, Path]]:
    result = []
    for split in ("train", "valid", "test"):
        candidates = [DATASET_ROOT / f"nsynth-{split}", DATASET_ROOT / split]
        directory = next((candidate for candidate in candidates if candidate.exists()), None)
        if directory is not None:
            result.append((split, directory))
    return result


def build_examples(task: str) -> tuple[list[ClipExample], int, dict[str, int]]:
    if task not in {"pitch", "family"}:
        raise ValueError("NSynth task must be 'pitch' or 'family'.")
    examples: list[ClipExample] = []
    label_names: dict[str, int] = {}
    for split, directory in _split_directories():
        metadata_files = list(directory.rglob("examples.json"))
        if not metadata_files:
            raise FileNotFoundError(f"examples.json not found under {directory}")
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        audio_root = metadata_files[0].parent / "audio"
        audio_map = {path.stem: path for path in audio_root.glob("*.wav")}
        if not audio_map:
            audio_map = {path.stem: path for path in directory.rglob("*.wav")}
        for key, item in metadata.items():
            audio_path = audio_map.get(key)
            if audio_path is None:
                continue
            if task == "pitch":
                # NSynth covers MIDI pitches 9--120 in this release. Build a
                # contiguous mapping from the metadata instead of assuming a
                # fixed piano range.
                label = int(item["pitch"])
            else:
                family = str(item.get("instrument_family_str", item["instrument_family"]))
                if family not in label_names:
                    label_names[family] = len(label_names)
                label = label_names[family]
            examples.append(ClipExample(audio_path, label, split, key))
    if task == "pitch":
        pitch_values = sorted({int(example.label) for example in examples})
        pitch_to_label = {pitch: index for index, pitch in enumerate(pitch_values)}
        examples = [
            ClipExample(
                example.path,
                pitch_to_label[int(example.label)],
                example.split,
                example.key,
                example.offset_seconds,
            )
            for example in examples
        ]
        label_names = {str(pitch): index for pitch, index in pitch_to_label.items()}
    return examples, len(label_names), label_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["pitch", "family"], default="pitch")
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=0, help="Maximum examples per split; 0 means all.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    examples, output_dim, label_names = build_examples(args.task)
    output = args.output or Path("probes/nsynth/results") / f"{args.task}.json"
    result = run_clip_probe(
        examples,
        checkpoint=args.checkpoint,
        task="multiclass",
        output_dim=output_dim,
        duration_seconds=4.0,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_items=args.max_items,
        seed=args.seed,
        device=args.device,
        output=output,
        random_init=args.random_init,
        feature_cache=args.feature_cache,
        profile=args.profile,
        progress=not args.no_progress,
        audio_workers=args.audio_workers,
        head_batch_size=args.head_batch_size,
    )
    result["labels"] = label_names
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
