"""Run a clip-level key-classification probe on GiantSteps Key."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from probes.common.runner import ClipExample, add_model_arguments, run_clip_probe
from probes.common.splits import hash_split

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "evaluation" / "giantsteps_key"
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


def _canonical_key(value: str) -> str | None:
    text = value.strip().replace("♯", "#").replace("♭", "b")
    match = re.search(r"([A-Ga-g](?:#|b)?)\s*(?::|/|-|\s)\s*(major|minor|maj|min|m)\b", text, re.IGNORECASE)
    if match is None:
        return None
    root = match.group(1).upper()
    root_id = ROOT_TO_ID.get(root)
    if root_id is None:
        return None
    mode = "min" if match.group(2).lower() in {"minor", "min", "m"} else "maj"
    return f"{root_id}:{mode}"


def _key_from_text(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        label = _canonical_key(line)
        if label is not None:
            return label
    return None


def _read_annotations(repository: Path) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for path in repository.rglob("*.key"):
        label = _key_from_text(path)
        if label is not None:
            annotations[path.stem] = label
    for path in repository.rglob("*.csv"):
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        for row in rows:
            key_column = next((name for name in row if name.lower() in {"key", "tonality", "annotation"}), None)
            id_column = next(
                (name for name in row if name.lower() in {"track_id", "track", "id", "filename", "file"}), None
            )
            if key_column is None or id_column is None:
                continue
            label = _canonical_key(row.get(key_column, ""))
            track_id = Path(row.get(id_column, "")).stem
            if label is not None and track_id:
                annotations[track_id] = label
    return annotations


def build_examples() -> tuple[list[ClipExample], dict[str, int]]:
    repositories = list(DATASET_ROOT.glob("giantsteps-key-dataset"))
    repository = repositories[0] if repositories else DATASET_ROOT
    annotations = _read_annotations(repository)
    audio_map = {path.stem: path for path in repository.rglob("*.mp3")}
    labels = sorted(set(annotations.values()))
    label_names = {label: index for index, label in enumerate(labels)}
    examples = [
        ClipExample(
            path=audio_map[key],
            label=label_names[label],
            split=hash_split(key),
            key=key,
        )
        for key, label in annotations.items()
        if key in audio_map
    ]
    if not examples:
        raise RuntimeError("No GiantSteps examples found; download audio with scripts/download_probe_datasets.py.")
    return examples, label_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("probes/giantsteps_key/results/key.json"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    examples, label_names = build_examples()
    result = run_clip_probe(
        examples,
        checkpoint=args.checkpoint,
        task="multiclass",
        output_dim=len(label_names),
        duration_seconds=30.0,
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
