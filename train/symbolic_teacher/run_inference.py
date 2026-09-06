"""Run the reference Tsumugi inference pipeline for one or more audio files.

Run this helper in the ``instrument_agnostic_amt`` Python environment, which
provides the transcription dependencies. Dataset preparation launches it as a
separate process and tokenizes the resulting MIDI in the Tsumugi-MRL environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

DEFAULT_MIDI_RESOLUTION = 1920
# pretty_midi refuses files whose largest tick is >= 10 million and allocates
# one entry per tick while parsing. Keep generated long-form files comfortably
# below that limit while retaining the normal AMT resolution for short audio.
SAFE_MIDI_MAX_TICK = 8_000_000
MIN_MIDI_RESOLUTION = 96
UNKNOWN_DURATION_MIDI_RESOLUTION = 480


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amt-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--task-file",
        type=Path,
        help="JSON list of {audio, result_json} tasks to process in one process.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-batch-size", type=int, default=2)
    parser.add_argument("--stem-splitter-batch-size", type=int, default=2)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--semi-crf-backend",
        choices=("triton", "torch", "auto"),
        default="triton",
    )
    parser.add_argument(
        "--midi-resolution",
        type=int,
        default=None,
        help=(
            "Override MIDI ticks per beat. By default 1920 is used, but it is "
            "lowered automatically for long audio so pretty_midi can parse it."
        ),
    )
    parser.set_defaults(
        refine_instruments=True,
        predict_beat_chord=True,
        compile_model=True,
    )
    parser.add_argument(
        "--no-refine-instruments",
        action="store_false",
        dest="refine_instruments",
        help="Disable the extra instrument refinement pass.",
    )
    parser.add_argument(
        "--no-predict-beat-chord",
        action="store_false",
        dest="predict_beat_chord",
        help="Disable beat/chord/key prediction.",
    )
    parser.add_argument(
        "--no-compile-model",
        action="store_false",
        dest="compile_model",
        help="Disable regional torch.compile.",
    )
    parser.add_argument(
        "--keep-separated-stems",
        action="store_true",
        help="Keep intermediate separated stem WAV files.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the task batch at the first inference error.",
    )
    return parser


def _resolve_midi_resolution(
    duration_seconds: float | None,
    requested_resolution: int | None,
) -> int:
    """Choose a PPQ that keeps a generated 120-BPM MIDI below pretty_midi's limit."""
    if requested_resolution is not None:
        if requested_resolution < MIN_MIDI_RESOLUTION:
            raise ValueError(f"--midi-resolution must be >= {MIN_MIDI_RESOLUTION}, got {requested_resolution}")
        return requested_resolution

    if duration_seconds is None or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        # Some compressed formats are not readable by soundfile. A conservative
        # PPQ keeps those files parseable without loading the entire audio just
        # to discover its duration.
        return UNKNOWN_DURATION_MIDI_RESOLUTION

    estimated_ticks = duration_seconds * 2.0 * DEFAULT_MIDI_RESOLUTION
    if estimated_ticks <= SAFE_MIDI_MAX_TICK:
        return DEFAULT_MIDI_RESOLUTION

    resolution = int(SAFE_MIDI_MAX_TICK / (duration_seconds * 2.0))
    return max(MIN_MIDI_RESOLUTION, min(DEFAULT_MIDI_RESOLUTION, resolution))


def _audio_duration_seconds(infer_stem: object, audio_path: Path) -> float | None:
    """Read duration without decoding the complete waveform."""
    soundfile_module = getattr(infer_stem, "sf", None)
    info = getattr(soundfile_module, "info", None)
    if info is None:
        return None
    try:
        return float(info(str(audio_path)).duration)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


@contextmanager
def _temporary_midi_resolution(pretty_midi_module: object, resolution: int):
    """Use a smaller resolution only when AMT creates a new PrettyMIDI object.

    MIDI files already on disk keep their own resolution while being read. The
    wrapper therefore changes only the constructor used by AMT's ``build_midi``
    and by later post-processing outputs, avoiding a global tick rescale.
    """
    original_class = pretty_midi_module.PrettyMIDI
    submodule = getattr(pretty_midi_module, "pretty_midi", None)
    original_submodule_class = getattr(submodule, "PrettyMIDI", None)

    class ResolvedPrettyMIDI(original_class):
        def __init__(self, *args, **kwargs):
            reading_existing_file = bool(args) or any(key in kwargs for key in ("midi_file", "mido_object"))
            if not reading_existing_file:
                kwargs["resolution"] = resolution
            super().__init__(*args, **kwargs)

    pretty_midi_module.PrettyMIDI = ResolvedPrettyMIDI
    if submodule is not None and original_submodule_class is not None:
        submodule.PrettyMIDI = ResolvedPrettyMIDI
    try:
        yield
    finally:
        pretty_midi_module.PrettyMIDI = original_class
        if submodule is not None and original_submodule_class is not None:
            submodule.PrettyMIDI = original_submodule_class


def _needs_safe_audio_alias(audio_path: Path) -> bool:
    """Return whether AMT's stem-based output directory would be invalid on Windows."""
    return sys.platform == "win32" and audio_path.stem != audio_path.stem.rstrip(" .")


def _stage_audio_alias(audio_path: Path, staging_root: Path, task_index: int) -> Path:
    """Create a safe-named hard link, falling back to a temporary copy."""
    suffix = audio_path.suffix or ".wav"
    staged_path = staging_root / f"audio_{task_index:06d}{suffix}"
    try:
        os.link(audio_path, staged_path)
    except OSError:
        # A hard link can fail across filesystems; copying is only the fallback
        # for filenames that otherwise make AMT's output path invalid.
        shutil.copyfile(audio_path, staged_path)
    return staged_path


def main() -> None:
    args = build_parser().parse_args()
    amt_root = args.amt_root.resolve()
    if not amt_root.is_dir():
        raise FileNotFoundError(f"instrument_agnostic_amt was not found: {amt_root}")
    sys.path.insert(0, str(amt_root))

    try:
        from instrument_agnostic_amt.cli import infer_stem
    except ImportError as exc:
        raise ImportError(
            "Could not import instrument_agnostic_amt inference. Run this helper "
            "with that repository's environment (the parent script detects its .venv)."
        ) from exc

    if args.compile_model and args.refine_instruments:
        original_get_refinement_models = infer_stem.get_refinement_models
        compiled_refinement_models: set[int] = set()

        def get_refinement_models(*model_args, **model_kwargs):
            bundle = original_get_refinement_models(*model_args, **model_kwargs)
            model = bundle["refinement_model"]
            if id(model) not in compiled_refinement_models:
                print("Compiling Instrument Refinement model ...", flush=True)
                infer_stem.maybe_compile_forward(
                    model,
                    enabled=True,
                    mode=args.compile_mode,
                )
                compiled_refinement_models.add(id(model))
            return bundle

        infer_stem.get_refinement_models = get_refinement_models

    if args.task_file is not None:
        tasks = json.loads(args.task_file.read_text(encoding="utf-8"))
    else:
        if args.audio is None or args.result_json is None:
            raise ValueError("--audio and --result-json are required without --task-file")
        tasks = [{"audio": str(args.audio), "result_json": str(args.result_json)}]

    output_root = args.output_root.resolve()
    staging_context = None
    if any(_needs_safe_audio_alias(Path(task["audio"])) for task in tasks):
        output_root.mkdir(parents=True, exist_ok=True)
        staging_context = tempfile.TemporaryDirectory(prefix=".safe_audio_", dir=output_root)

    try:
        staging_root = Path(staging_context.name) if staging_context is not None else None
        for index, task in enumerate(tasks, start=1):
            audio_path = Path(task["audio"]).resolve()
            result_json = Path(task["result_json"]).resolve()
            staged_audio_path = None
            if staging_root is not None and _needs_safe_audio_alias(audio_path):
                staged_audio_path = _stage_audio_alias(audio_path, staging_root, index)

            inference_audio_path = staged_audio_path or audio_path
            print(f"\n[{index}/{len(tasks)}] Tsumugi: {audio_path.name}", flush=True)
            try:
                duration_seconds = _audio_duration_seconds(infer_stem, inference_audio_path)
                midi_resolution = _resolve_midi_resolution(
                    duration_seconds,
                    args.midi_resolution,
                )
                if midi_resolution != DEFAULT_MIDI_RESOLUTION:
                    duration_text = (
                        f"{duration_seconds / 60.0:.1f} min" if duration_seconds is not None else "unknown duration"
                    )
                    print(
                        f"Using MIDI resolution {midi_resolution} for {duration_text} "
                        "audio to stay below pretty_midi's tick limit.",
                        flush=True,
                    )
                midi_context = (
                    _temporary_midi_resolution(infer_stem.pretty_midi, midi_resolution)
                    if midi_resolution != DEFAULT_MIDI_RESOLUTION
                    else nullcontext()
                )
                with midi_context:
                    result = infer_stem.run_stem_separated_transcription(
                        inference_audio_path,
                        output_root=output_root,
                        stem_splitter_batch_size=args.stem_splitter_batch_size,
                        window_batch_size=args.window_batch_size,
                        cleanup_separated_stems=not args.keep_separated_stems,
                        max_midi_melodic_instruments=15,
                        transcribe_drum_stems=True,
                        refine_instruments=args.refine_instruments,
                        refinement_mode="cluster",
                        predict_velocity=False,
                        predict_beat_chord=args.predict_beat_chord,
                        beat_chord_use_audio=True,
                        device=args.device,
                        amp=True,
                        compile_model=args.compile_model,
                        compile_velocity=False,
                        compile_mode=args.compile_mode,
                        semi_crf_backend=args.semi_crf_backend,
                    )
            except Exception as exc:
                failure = {
                    "status": "error",
                    "audio_path": str(audio_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                result_json.parent.mkdir(parents=True, exist_ok=True)
                result_json.write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"Tsumugi failed for {audio_path.name}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
                continue
            finally:
                if staged_audio_path is not None:
                    staged_audio_path.unlink(missing_ok=True)

            # Keep the source metadata independent of the temporary alias.
            result["audio_path"] = str(audio_path)
            result["midi_resolution"] = midi_resolution
            result_json.parent.mkdir(parents=True, exist_ok=True)
            result_json.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Tsumugi result: {result_json}", flush=True)
    finally:
        if staging_context is not None:
            staging_context.cleanup()


if __name__ == "__main__":
    main()
