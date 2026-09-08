"""Run the OpenMIC-2018 multi-label instrument probe."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from probes.common.runner import ClipExample, add_model_arguments, run_clip_probe

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "evaluation" / "openmic"


def _find(name: str) -> Path:
    candidates = list(DATASET_ROOT.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"{name} not found below {DATASET_ROOT}")
    return candidates[0]


def _read_partition() -> dict[str, str]:
    partition_dir = DATASET_ROOT / "openmic-2018" / "partitions"
    if not partition_dir.exists():
        candidates = list(DATASET_ROOT.rglob("train01.txt"))
        partition_dir = candidates[0].parent if candidates else partition_dir
    train_file = partition_dir / "train01.txt"
    test_file = partition_dir / "test01.txt"
    if train_file.exists() and test_file.exists():
        partitions: dict[str, str] = {}
        for path, split in ((train_file, "train"), (test_file, "test")):
            for line in path.read_text(encoding="utf-8").splitlines():
                sample_key = line.strip()
                if sample_key:
                    partitions[sample_key] = split
        return partitions

    split_train_file = partition_dir / "split01_train.csv"
    split_test_file = partition_dir / "split01_test.csv"
    if split_train_file.exists() and split_test_file.exists():
        partitions = {}
        for path, split in ((split_train_file, "train"), (split_test_file, "test")):
            for line in path.read_text(encoding="utf-8").splitlines():
                sample_key = line.strip().split(",", 1)[0]
                if sample_key:
                    partitions[sample_key] = split
        return partitions

    candidates = list(DATASET_ROOT.rglob("split01.csv")) + list(DATASET_ROOT.rglob("*.csv"))
    for path in candidates:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows or "sample_key" not in rows[0]:
            continue
        split_column = next((column for column in ("subset", "split", "partition") if column in rows[0]), None)
        if split_column is None:
            continue
        return {row["sample_key"].strip(): row[split_column].strip().lower() for row in rows}
    raise FileNotFoundError("An OpenMIC partition CSV was not found.")


def build_examples() -> tuple[list[ClipExample], dict[str, int]]:
    labels_file = _find("openmic-2018-aggregated-labels.csv")
    with labels_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("OpenMIC label CSV is empty.")
    instruments = sorted({row["instrument"].strip() for row in rows if row.get("instrument", "").strip()})
    label_names = {instrument: index for index, instrument in enumerate(instruments)}
    labels: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(instruments))
    for row in rows:
        sample_key = row.get("sample_key", "").strip()
        instrument = row.get("instrument", "").strip()
        try:
            likelihood = float(row.get("likelihood", row.get("relevance", row.get("confidence", "0"))))
        except ValueError:
            likelihood = 0.0
        if instrument in label_names and likelihood >= 0.5:
            labels[sample_key][label_names[instrument]] = 1.0

    partitions = _read_partition()
    audio_map = {path.stem: path for path in DATASET_ROOT.rglob("*.ogg")}
    audio_map.update({path.stem: path for path in DATASET_ROOT.rglob("*.wav")})
    examples: list[ClipExample] = []
    split_aliases = {"training": "train", "train": "train", "validation": "valid", "valid": "valid", "test": "test"}
    for sample_key, label in labels.items():
        audio_path = audio_map.get(sample_key)
        raw_split = partitions.get(sample_key, "")
        if audio_path is None or raw_split not in split_aliases:
            continue
        examples.append(ClipExample(audio_path, label, split_aliases[raw_split], sample_key))
    if not examples:
        raise RuntimeError("No OpenMIC examples found; check the extracted archive layout.")
    return examples, label_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("probes/openmic/results/instrument.json"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    examples, label_names = build_examples()
    result = run_clip_probe(
        examples,
        checkpoint=args.checkpoint,
        task="multilabel",
        output_dim=len(label_names),
        duration_seconds=10.0,
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
