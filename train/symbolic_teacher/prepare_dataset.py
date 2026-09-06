"""Create Tsumugi MIDI and FRAME-anchored symbolic-token caches.

This script runs the normal Tsumugi pipeline with instrument refinement,
beat/chord/key prediction, and regional ``torch.compile`` enabled by default.
All pending audio files are handled by one AMT process so its models can be
reused. The resulting MIDI is then tokenized by ``MIDIEventTokenizer``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch

# Direct execution by file path needs the repository root on sys.path
# so the train package can be imported without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.symbolic import MIDIEventTokenizer, SYMBOLIC_CACHE_VERSION


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
PROGRESS_INTERVAL_SECONDS = 10.0


def _safe_name(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    value = "__".join(relative.with_suffix("").parts)
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return value or "audio"


def _has_completed_inference(result_json: Path) -> bool:
    """Return whether a result JSON points to a materialized merged MIDI."""
    result = _read_inference_result(result_json)
    merged_midi_path = result.get("merged_midi_path") if result is not None else None
    return isinstance(merged_midi_path, str) and Path(merged_midi_path).is_file()


def _read_inference_result(result_json: Path) -> dict[str, object] | None:
    if not result_json.is_file():
        return None
    try:
        value = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_failed_inference(result_json: Path) -> bool:
    result = _read_inference_result(result_json)
    return bool(result and result.get("status") == "error")


def _resolve_python(amt_root: Path, requested: str | None) -> list[str]:
    if requested:
        return [requested]
    project_root = Path(__file__).resolve().parents[2]
    if sys.platform == "win32":
        candidates = (
            project_root / ".venv-amt" / "Scripts" / "python.exe",
            amt_root / ".venv" / "Scripts" / "python.exe",
        )
    else:
        candidates = (
            project_root / ".venv-amt" / "bin" / "python",
            amt_root / ".venv" / "bin" / "python",
        )
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]
    # The AMT checkout has its own optional `stem` dependencies. Let uv create
    # or reuse that environment when no platform-native interpreter is present.
    return ["uv", "run", "--project", str(amt_root), "--extra", "stem", "python"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path("datasets/audios"))
    parser.add_argument("--amt-root", type=Path, default=Path("../instrument_agnostic_amt"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/symbolic"))
    parser.add_argument("--amt-python", default=None, help="Python executable for instrument_agnostic_amt.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-batch-size", type=int, default=2)
    parser.add_argument("--stem-splitter-batch-size", type=int, default=2)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--midi-resolution",
        type=int,
        default=None,
        help="Override MIDI ticks per beat; long audio is lowered automatically by default.",
    )
    parser.add_argument(
        "--semi-crf-backend",
        choices=("triton", "torch", "auto"),
        default="triton",
    )
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry inference results recorded with status=error.",
    )
    parser.add_argument(
        "--retokenize",
        action="store_true",
        help="Rebuild caches from output-dir/midi only; do not run audio inference.",
    )
    return parser


def _save_token_cache(tokenizer: MIDIEventTokenizer, midi_path: Path, token_path: Path) -> dict[str, object]:
    sequence = tokenizer.encode(midi_path)
    payload = {
        "cache_version": SYMBOLIC_CACHE_VERSION,
        "token_ids": sequence.token_ids,
        "token_type_ids": sequence.token_type_ids,
        "token_instrument_ids": sequence.token_instrument_ids,
        "note_spans": sequence.note_spans,
        "position_ids": sequence.position_ids,
        "anchor_positions": sequence.anchor_positions,
        "frame_targets": sequence.frame_targets.as_dict(),
        "duration_seconds": sequence.duration_seconds,
    }
    token_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, token_path)
    return {
        "midi_path": str(midi_path),
        "token_path": str(token_path),
        "num_tokens": int(sequence.token_ids.numel()),
        "num_frames": int(sequence.num_frames),
        "duration_seconds": float(sequence.duration_seconds),
    }


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    return str(timedelta(seconds=max(0, int(seconds))))


def _print_progress(label: str, completed: int, total: int, started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    rate = completed / elapsed if completed and elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate else None
    percent = completed / total * 100 if total else 100.0
    print(
        f"{label}: {completed}/{total} ({percent:.1f}%) | "
        f"elapsed {_format_time(elapsed)} | ETA {_format_time(remaining)} | "
        f"{rate * 60:.2f} files/min",
        flush=True,
    )


def _run_inference_with_progress(
    command: list[str],
    amt_root: Path,
    pending_tasks: list[dict[str, str]],
) -> None:
    result_paths = [Path(task["result_json"]) for task in pending_tasks]
    initial_mtimes = {path: path.stat().st_mtime_ns for path in result_paths if path.is_file()}

    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    ffmpeg_bin = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg" / "bin"
    if ffmpeg_bin.is_dir():
        child_env["PATH"] = os.pathsep.join((str(ffmpeg_bin), child_env["PATH"]))
    process = subprocess.Popen(command, cwd=str(amt_root), env=child_env)
    started_at = time.monotonic()
    last_completed = -1
    last_report_at = started_at
    try:
        while process.poll() is None:
            completed = sum(
                path.is_file() and (path not in initial_mtimes or path.stat().st_mtime_ns > initial_mtimes[path])
                for path in result_paths
            )
            now = time.monotonic()
            if completed != last_completed or now - last_report_at >= PROGRESS_INTERVAL_SECONDS:
                _print_progress("Tsumugi inference", completed, len(pending_tasks), started_at)
                last_completed = completed
                last_report_at = now
            time.sleep(2.0)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise

    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    _print_progress("Tsumugi inference", len(pending_tasks), len(pending_tasks), started_at)


def main() -> None:
    args = build_parser().parse_args()
    audio_dir = args.audio_dir.resolve()
    amt_root = args.amt_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.retokenize:
        midi_paths = sorted((output_dir / "midi").glob("*.mid"))
        if not midi_paths:
            raise FileNotFoundError(f"No prepared MIDI files under {output_dir / 'midi'}")
        tokenizer = MIDIEventTokenizer(frame_rate=args.frame_rate)
        metadata_by_path = {}
        for midi_path in midi_paths:
            metadata = _save_token_cache(tokenizer, midi_path, output_dir / "tokens" / f"{midi_path.stem}.pt")
            metadata_by_path[str(midi_path)] = metadata
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest:
                midi_value = entry.get("midi_path") if isinstance(entry, dict) else None
                if isinstance(midi_value, str):
                    entry.update(metadata_by_path.get(str(Path(midi_value).resolve()), {}))
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rebuilt {len(midi_paths)} paired-event caches from existing MIDI; no audio inference.")
        return
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not amt_root.is_dir():
        raise FileNotFoundError(f"instrument_agnostic_amt was not found: {amt_root}")
    if args.window_batch_size <= 0 or args.stem_splitter_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    midi_dir = output_dir / "midi"
    token_dir = output_dir / "tokens"
    inference_dir = output_dir / "inference"
    for directory in (midi_dir, token_dir, inference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    child_script = Path(__file__).with_name("run_inference.py").resolve()
    amt_python = _resolve_python(amt_root, args.amt_python)
    tokenizer = MIDIEventTokenizer(frame_rate=args.frame_rate)
    audio_paths = sorted(
        path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_paths:
        raise FileNotFoundError(f"No audio files found under {audio_dir}")

    jobs = []
    pending_tasks = []
    for index, audio_path in enumerate(audio_paths):
        stem = f"{index:06d}_{_safe_name(audio_path, audio_dir)}"
        midi_path = midi_dir / f"{stem}.mid"
        token_path = token_dir / f"{stem}.pt"
        result_json = inference_dir / f"{stem}.json"
        cache_current = token_path.is_file() and (
            torch.load(token_path, map_location="cpu").get("cache_version") == SYMBOLIC_CACHE_VERSION
        )
        failed_inference_cached = _is_failed_inference(result_json)
        needs_tokenization = args.force or not cache_current or not midi_path.exists()
        if failed_inference_cached and not (args.force or args.retry_failed):
            # Preserve the failure in the manifest without trying to tokenize a
            # missing MIDI on every subsequent dataset-preparation run.
            needs_tokenization = False
        needs_inference = args.force or (
            (args.retry_failed and failed_inference_cached)
            or (needs_tokenization and not midi_path.is_file() and not _has_completed_inference(result_json))
        )
        jobs.append((audio_path, midi_path, token_path, result_json, needs_inference, needs_tokenization))
        if needs_inference:
            pending_tasks.append(
                {
                    "audio": str(audio_path),
                    "result_json": str(result_json),
                }
            )

    print(
        f"Found {len(jobs)} audio files: {len(pending_tasks)} pending inference, "
        f"{len(jobs) - len(pending_tasks)} inference cached",
        flush=True,
    )
    if pending_tasks:
        task_file = inference_dir / "pending_tasks.json"
        task_file.write_text(
            json.dumps(pending_tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = [
            *amt_python,
            str(child_script),
            "--amt-root",
            str(amt_root),
            "--task-file",
            str(task_file),
            "--output-root",
            str(inference_dir),
            "--device",
            str(args.device),
            "--window-batch-size",
            str(args.window_batch_size),
            "--stem-splitter-batch-size",
            str(args.stem_splitter_batch_size),
            "--compile-mode",
            str(args.compile_mode),
            "--semi-crf-backend",
            str(args.semi_crf_backend),
        ]
        if args.midi_resolution is not None:
            command.extend(("--midi-resolution", str(args.midi_resolution)))
        print(f"\nTsumugi: {len(pending_tasks)} files in one process", flush=True)
        _run_inference_with_progress(command, amt_root, pending_tasks)
        task_file.unlink()

    manifest: list[dict[str, object]] = []
    tokenization_started_at = time.monotonic()
    for index, (
        audio_path,
        midi_path,
        token_path,
        result_json,
        needs_inference,
        needs_tokenization,
    ) in enumerate(jobs, start=1):
        if needs_inference:
            if not result_json.is_file():
                raise RuntimeError(f"Tsumugi helper did not create {result_json}")
        inference_result = _read_inference_result(result_json)
        if inference_result and inference_result.get("status") == "error":
            failure_metadata = {
                "audio_path": str(audio_path),
                "status": "error",
                "error_type": str(inference_result.get("error_type", "Error")),
                "error": str(inference_result.get("error", "unknown inference error")),
            }
            manifest.append(failure_metadata)
            print(
                f"[{index}/{len(jobs)}] skipped {audio_path.name}: "
                f"{failure_metadata['error_type']}: {failure_metadata['error']}",
                flush=True,
            )
            _print_progress("Dataset preparation", index, len(jobs), tokenization_started_at)
            continue
        if needs_tokenization:
            if needs_inference or not midi_path.is_file():
                result = inference_result or {}
                source_midi = Path(result["merged_midi_path"])
                if not source_midi.is_file():
                    raise FileNotFoundError(f"Tsumugi output MIDI not found: {source_midi}")
                shutil.copyfile(source_midi, midi_path)
            metadata = _save_token_cache(tokenizer, midi_path, token_path)
        else:
            cache = torch.load(token_path, map_location="cpu")
            metadata = {
                "midi_path": str(midi_path),
                "token_path": str(token_path),
                "num_tokens": int(cache["token_ids"].numel()),
                "num_frames": int(cache["anchor_positions"].numel()),
                "duration_seconds": float(cache["duration_seconds"]),
            }

        manifest.append({"audio_path": str(audio_path), **metadata})
        print(
            f"[{index}/{len(jobs)}] saved MIDI={midi_path.name} "
            f"tokens={metadata['num_tokens']} frames={metadata['num_frames']}",
            flush=True,
        )
        _print_progress("Dataset preparation", index, len(jobs), tokenization_started_at)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
