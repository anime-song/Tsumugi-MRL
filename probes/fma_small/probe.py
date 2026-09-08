"""Run the official FMA-small genre split as a clip-level probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from probes.common.runner import ClipExample, add_model_arguments, run_clip_probe

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "evaluation" / "fma_small"

# These three files in the official FMA-small archive contain only ID3 tags
# and have no decodable MPEG frames.  Keep the exclusion explicit so the
# evaluation remains reproducible instead of failing part-way through.
KNOWN_INVALID_TRACK_IDS = {"099134", "108925", "133297"}


def _tracks_file() -> Path:
    candidates = list(DATASET_ROOT.rglob("tracks.csv"))
    if not candidates:
        raise FileNotFoundError(f"tracks.csv not found below {DATASET_ROOT}")
    return candidates[0]


def _column_names(first: list[str], second: list[str]) -> list[str]:
    names = []
    for parent, child in zip(first, second):
        parent = parent.strip()
        child = child.strip()
        if parent and child:
            names.append(f"{parent}.{child}")
        else:
            names.append(child or parent)
    # The FMA metadata reserves the first column for the numeric track ID and
    # leaves both header rows blank in that position.
    if names and not names[0]:
        names[0] = "track_id"
    return names


def build_examples() -> tuple[list[ClipExample], dict[str, int]]:
    tracks_file = _tracks_file()
    with tracks_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first = next(reader)
        second = next(reader)
        columns = _column_names(first, second)
        rows = [dict(zip(columns, row)) for row in reader]

    audio_map = {path.stem: path for path in DATASET_ROOT.rglob("*.mp3")}
    genres = sorted({row.get("track.genre_top", "").strip() for row in rows if row.get("track.genre_top", "").strip()})
    label_names = {genre: index for index, genre in enumerate(genres)}
    examples: list[ClipExample] = []
    for row in rows:
        track_id = row.get("track_id", "").strip()
        genre = row.get("track.genre_top", "").strip()
        split = row.get("set.split", "").strip()
        subset = row.get("set.subset", "").strip()
        if (
            not track_id.isdigit()
            or genre not in label_names
            or split not in {"training", "validation", "test", "train", "valid"}
        ):
            continue
        if subset and subset != "small":
            continue
        audio_path = audio_map.get(track_id.zfill(6))
        if audio_path is None:
            continue
        if audio_path.stem in KNOWN_INVALID_TRACK_IDS:
            continue
        normalized_split = {
            "training": "train",
            "train": "train",
            "validation": "valid",
            "valid": "valid",
            "test": "test",
        }[split]
        examples.append(ClipExample(audio_path, label_names[genre], normalized_split, track_id))
    if not examples:
        raise RuntimeError("No FMA-small examples found; check the extracted archive layout.")
    return examples, label_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("probes/fma_small/results/genre.json"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
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
    result["excluded_track_ids"] = sorted(KNOWN_INVALID_TRACK_IDS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
