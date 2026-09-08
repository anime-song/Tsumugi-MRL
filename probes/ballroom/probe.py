"""Run a frame-level beat-detection probe on Ballroom."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import soundfile as sf
import torch

from probes.common.runner import FrameExample, add_model_arguments, run_frame_probe
from probes.common.splits import hash_split

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "evaluation" / "ballroom"

# The annotation README identifies exact and recording replicas. Keep each
# listed pair in one split so the beat probe cannot benefit from an audio copy
# appearing in both train and test.
_REPLICA_PAIRS = [
    ("Albums-AnaBelen_Veneo-11", "Albums-Chrisanne2-12"),
    ("Albums-Fire-08", "Albums-Fire-09"),
    ("Albums-Latin_Jam2-05", "Albums-Latin_Jam2-13"),
    ("Albums-Secret_Garden-01", "Media-104705"),
    ("Albums-AnaBelen_Veneo-03", "Albums-AnaBelen_Veneo-15"),
    ("Albums-Ballroom_Magic-03", "Albums-Ballroom_Magic-18"),
    ("Albums-Latin_Jam-04", "Albums-Latin_Jam-13"),
    ("Albums-Latin_Jam-08", "Albums-Latin_Jam-14"),
    ("Albums-Latin_Jam-06", "Albums-Latin_Jam-15"),
    ("Albums-Latin_Jam2-02", "Albums-Latin_Jam2-14"),
    ("Albums-Latin_Jam2-07", "Albums-Latin_Jam2-15"),
    ("Albums-Latin_Jam3-02", "Media-103414"),
    ("Media-103402", "Media-103415"),
]
_REPLICA_GROUP = {name: f"replica-{index}" for index, pair in enumerate(_REPLICA_PAIRS) for name in pair}


def _read_beats(path: Path) -> list[float]:
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        try:
            values.append(float(fields[0]))
        except ValueError:
            continue
    return sorted(values)


def _label_function(beats: list[float]):
    def label_fn(times: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(times.numel(), dtype=torch.float32)
        for index, time in enumerate(times.tolist()):
            position = bisect_left(beats, time)
            nearest = min(
                (abs(beats[candidate] - time) for candidate in (position - 1, position) if 0 <= candidate < len(beats)),
                default=float("inf"),
            )
            # One 25-Hz frame spans roughly 40 ms; 60 ms tolerates the
            # center-time convention without turning adjacent frames positive.
            result[index] = float(nearest <= 0.06)
        return result

    return label_fn


def build_examples() -> list[FrameExample]:
    annotation_files = sorted(DATASET_ROOT.rglob("*.beats"))
    audio_map = {path.stem: path for path in DATASET_ROOT.rglob("*.wav")}
    if not annotation_files or not audio_map:
        raise FileNotFoundError(f"Ballroom audio/annotations not found below {DATASET_ROOT}")
    examples = []
    for annotation_file in annotation_files:
        audio_path = audio_map.get(annotation_file.stem)
        if audio_path is None:
            continue
        beats = _read_beats(annotation_file)
        if not beats:
            continue
        duration = min(30.0, float(sf.info(str(audio_path)).duration))
        examples.append(
            FrameExample(
                path=audio_path,
                split=hash_split(_REPLICA_GROUP.get(annotation_file.stem, annotation_file.stem)),
                key=annotation_file.stem,
                label_fn=_label_function(beats),
                duration_seconds=duration,
            )
        )
    if not examples:
        raise RuntimeError("No Ballroom examples found; check the extracted archive layout.")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("probes/ballroom/results/beat.json"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_frame_probe(
        build_examples(),
        checkpoint=args.checkpoint,
        task="binary",
        output_dim=1,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
