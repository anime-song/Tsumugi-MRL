"""Run a frame-level chord probe on GuitarSet's mic recordings."""

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from pathlib import Path

import torch

from probes.common.runner import FrameExample, add_model_arguments, run_frame_probe

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "evaluation" / "guitarset"
ROOT_TO_ID = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def _chord_class(value: object) -> str:
    text = str(value).strip()
    if not text or text.upper() in {"N", "NONE", "NO_CHORD"}:
        return "N"
    match = re.match(r"^([A-Ga-g](?:#|b)?)(?::|/)?(.*)$", text)
    if match is None:
        return "N"
    root = match.group(1).upper().replace("♯", "#").replace("♭", "B")
    root_id = ROOT_TO_ID.get(root)
    if root_id is None:
        return "N"
    suffix = match.group(2).lower()
    if "dim" in suffix:
        quality = "dim"
    elif "aug" in suffix:
        quality = "aug"
    elif "min" in suffix or (suffix.startswith("m") and not suffix.startswith("maj")):
        quality = "min"
    elif "maj" in suffix or suffix in {"", "major"}:
        quality = "maj"
    elif "7" in suffix:
        quality = "7"
    else:
        quality = "other"
    return f"{root_id}:{quality}"


def _chord_intervals(annotation_file: Path) -> list[tuple[float, float, str]]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    candidates = [annotation for annotation in data.get("annotations", []) if annotation.get("namespace") == "chord"]
    if not candidates:
        return []
    intervals = []
    for item in candidates[0].get("data", []):
        try:
            start = float(item["time"])
            duration = float(item.get("duration", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        intervals.append((start, start + max(0.0, duration), _chord_class(item.get("value", "N"))))
    return sorted(intervals)


def _annotation_duration(annotation_file: Path) -> float:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    try:
        return min(30.0, float(data["file_metadata"]["duration"]))
    except (KeyError, TypeError, ValueError):
        return 30.0


def _label_function(intervals: list[tuple[float, float, str]], label_ids: dict[str, int]):
    starts = [interval[0] for interval in intervals]
    no_chord = label_ids["N"]

    def label_fn(times: torch.Tensor) -> torch.Tensor:
        result = torch.full((times.numel(),), no_chord, dtype=torch.long)
        for index, time in enumerate(times.tolist()):
            interval_index = bisect_right(starts, time) - 1
            if interval_index >= 0:
                start, end, label = intervals[interval_index]
                if start <= time < end:
                    result[index] = label_ids[label]
        return result

    return label_fn


def _find_audio(annotation_file: Path, audio_map: dict[str, Path]) -> Path | None:
    stem = annotation_file.stem
    for candidate in (stem, f"{stem}_mic", stem.replace("_jams", "_mic")):
        if candidate in audio_map:
            return audio_map[candidate]
    matches = [path for key, path in audio_map.items() if key.startswith(stem) and "mic" in key]
    return matches[0] if matches else None


def build_examples() -> tuple[list[FrameExample], dict[str, int]]:
    annotation_files = sorted(DATASET_ROOT.rglob("*.jams"))
    if not annotation_files:
        raise FileNotFoundError(f"No JAMS annotations found below {DATASET_ROOT}")
    audio_map = {path.stem: path for path in DATASET_ROOT.rglob("*.wav") if "mic" in path.stem.lower()}
    if not audio_map:
        audio_map = {path.stem: path for path in DATASET_ROOT.rglob("*.wav")}

    parsed: list[tuple[Path, Path, list[tuple[float, float, str]], str, float]] = []
    labels = {"N"}
    for annotation_file in annotation_files:
        intervals = _chord_intervals(annotation_file)
        audio_path = _find_audio(annotation_file, audio_map)
        if audio_path is None or not intervals:
            continue
        labels.update(interval[2] for interval in intervals)
        group = annotation_file.stem.split("_")[0]
        parsed.append((annotation_file, audio_path, intervals, group, _annotation_duration(annotation_file)))
    label_names = {label: index for index, label in enumerate(sorted(labels))}

    groups = sorted({item[3] for item in parsed})
    if len(groups) >= 3:
        train_end = max(1, len(groups) - 2)
        split_by_group = {
            group: ("train" if index < train_end else "valid" if index == train_end else "test")
            for index, group in enumerate(groups)
        }
    else:
        split_by_group = {group: "train" for group in groups}
    examples = [
        FrameExample(
            path=audio_path,
            split=split_by_group[group],
            key=annotation_file.stem,
            label_fn=_label_function(intervals, label_names),
            duration_seconds=duration,
        )
        for annotation_file, audio_path, intervals, group, duration in parsed
    ]
    if not examples or "test" not in {example.split for example in examples}:
        raise RuntimeError("GuitarSet examples could not be split into train/valid/test.")
    return examples, label_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("probes/guitarset/results/chord.json"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    examples, label_names = build_examples()
    result = run_frame_probe(
        examples,
        checkpoint=args.checkpoint,
        task="multiclass",
        output_dim=len(label_names),
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_items=args.max_items,
        seed=args.seed,
        device=args.device,
        output=args.output,
        random_init=args.random_init,
        feature_cache=args.feature_cache,
        profile=args.profile,
        progress=not args.no_progress,
        audio_workers=args.audio_workers,
        head_batch_size=args.head_batch_size,
    )
    result["labels"] = label_names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
