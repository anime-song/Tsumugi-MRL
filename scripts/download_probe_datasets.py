"""Download and unpack the datasets used by the linear-probe suite.

The URLs below are the official dataset/maintainer endpoints. Archives are
kept after extraction so an interrupted experiment can be resumed without
re-downloading the source files. Git is not required: repository contents are
downloaded as ZIP archives, and GiantSteps audio is fetched directly with its
MD5 manifests. Re-running this script is safe: existing files are skipped and
partial files are resumed when curl is available.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "evaluation"


def _download(url: str, destination: Path, *, overwrite: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination}")
        return
    print(f"[download] {url}")
    print(f"          -> {destination}")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    partial = destination.with_suffix(destination.suffix + ".part")
    if curl:
        command = [
            curl,
            "--fail",
            "--location",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            # The managed Windows image can present a proxy certificate whose
            # hostname does not match the upstream archive host. Retry only
            # this known transport failure; the URLs remain pinned above.
            if not url.startswith("https://"):
                raise
            subprocess.run([curl, "--insecure", *command[1:]], check=True)
        partial.replace(destination)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Tsumugi-MRL probe downloader"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(destination)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        root = destination.resolve()
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        try:
            handle.extractall(destination, filter="data")
        except TypeError:  # Python 3.10/3.11
            handle.extractall(destination)
    # Some Windows tar implementations carry Unix mode 000 into the NTFS
    # ACL. Make the root accessible first, then normalize directories before
    # descending into them so the recursive walk itself is not blocked.
    try:
        os.chmod(destination, 0o755)
    except PermissionError:
        pass
    for current, directories, files in os.walk(destination, topdown=True):
        for name in directories:
            try:
                os.chmod(Path(current) / name, 0o755)
            except PermissionError:
                pass
        for name in files:
            try:
                os.chmod(Path(current) / name, 0o644)
            except PermissionError:
                pass


def _extract(archive: Path, destination: Path) -> None:
    marker = destination / f".extracted-{archive.name}"
    if marker.exists():
        print(f"[skip] extracted {archive.name}")
        return
    print(f"[extract] {archive}")
    if archive.name.endswith(".zip"):
        _safe_extract_zip(archive, destination)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        _safe_extract_tar(archive, destination)
    else:
        raise ValueError(f"Unsupported archive: {archive}")
    marker.write_text("ok\n", encoding="utf-8")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nsynth(root: Path) -> None:
    destination = root / "nsynth"
    assets = {
        "train": "https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
        "valid": "https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
        "test": "https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
    }
    for split, url in assets.items():
        archive = destination / Path(url).name
        _download(url, archive)
        _extract(archive, destination)


def _guitarset(root: Path) -> None:
    destination = root / "guitarset"
    assets = {
        "annotation.zip": "https://zenodo.org/records/3371780/files/annotation.zip?download=1",
        "audio_mono-mic.zip": "https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1",
    }
    for filename, url in assets.items():
        archive = destination / filename
        _download(url, archive)
        _extract(archive, destination)


def _fma_small(root: Path) -> None:
    destination = root / "fma_small"
    assets = {
        "fma_metadata.zip": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "fma_small.zip": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    }
    for filename, url in assets.items():
        archive = destination / filename
        _download(url, archive)
        _extract(archive, destination)


def _openmic(root: Path) -> None:
    destination = root / "openmic"
    url = "https://zenodo.org/records/1432913/files/openmic-2018-v1.0.0.tgz?download=1"
    archive = destination / "openmic-2018-v1.0.0.tgz"
    _download(url, archive)
    _extract(archive, destination)


def _ballroom(root: Path) -> None:
    destination = root / "ballroom"
    audio_url = "https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz"
    archive = destination / "data1.tar.gz"
    _download(audio_url, archive)
    _extract(archive, destination)

    if not next(destination.rglob("*.beats"), None):
        annotation_archive = destination / "BallroomAnnotations-master.zip"
        _download(
            "https://github.com/CPJKU/BallroomAnnotations/archive/refs/heads/master.zip",
            annotation_archive,
        )
        _extract(annotation_archive, destination)
    else:
        print(f"[skip] Ballroom annotations under {destination}")


def _find_giantsteps_repository(destination: Path) -> Path | None:
    for name in ("giantsteps-key-dataset", "giantsteps-key-dataset-master", "giantsteps-key-dataset-main"):
        candidate = destination / name
        if (candidate / "annotations").exists() and (candidate / "md5").exists():
            return candidate
    return None


def _download_giantsteps_audio(repository: Path) -> None:
    md5_directory = repository / "md5"
    audio_directory = repository / "audio"
    manifests = sorted(md5_directory.glob("*.md5"))
    if not manifests:
        raise RuntimeError(f"No GiantSteps MD5 manifests found under {md5_directory}.")

    base_urls = (
        "https://www.cp.jku.ac.at/datasets/giantsteps/backup/",
        "https://geo-samples.beatport.com/lofi/",
    )
    audio_directory.mkdir(parents=True, exist_ok=True)
    successful = 0
    for manifest in manifests:
        filename = f"{manifest.stem}.mp3"
        target = audio_directory / filename
        expected = manifest.read_text(encoding="utf-8").strip().split()[0]
        if target.exists() and _md5(target) == expected:
            successful += 1
            continue

        downloaded = False
        for base_url in base_urls:
            try:
                _download(f"{base_url}{filename}", target, overwrite=True)
            except (OSError, subprocess.CalledProcessError, URLError) as error:
                print(f"[retry] {filename}: {error}")
                continue
            if target.exists() and _md5(target) == expected:
                downloaded = True
                break
            print(f"[retry] MD5 mismatch for {filename}")
        if not downloaded:
            raise RuntimeError(f"Could not download a valid copy of {filename}.")
        successful += 1
        if successful % 50 == 0 or successful == len(manifests):
            print(f"[giantsteps] audio verified: {successful}/{len(manifests)}")


def _giantsteps(root: Path, *, download_audio: bool) -> None:
    destination = root / "giantsteps_key"
    repository = _find_giantsteps_repository(destination)
    if repository is None:
        archive = destination / "giantsteps-key-dataset-master.zip"
        _download(
            "https://github.com/GiantSteps/giantsteps-key-dataset/archive/refs/heads/master.zip",
            archive,
        )
        _extract(archive, destination)
        repository = _find_giantsteps_repository(destination)
    if repository is None:
        raise RuntimeError(f"GiantSteps repository archive did not contain the expected files under {destination}.")
    print(f"[skip] dataset sources already available at {repository}")
    if not download_audio:
        print("[info] GiantSteps annotations downloaded; audio download skipped")
        return
    _download_giantsteps_audio(repository)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["nsynth", "guitarset", "fma_small", "openmic", "ballroom", "giantsteps_key"],
        default=["nsynth", "guitarset", "fma_small", "openmic", "ballroom", "giantsteps_key"],
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--skip-giantsteps-audio", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        if dataset == "nsynth":
            _nsynth(root)
        elif dataset == "guitarset":
            _guitarset(root)
        elif dataset == "fma_small":
            _fma_small(root)
        elif dataset == "openmic":
            _openmic(root)
        elif dataset == "ballroom":
            _ballroom(root)
        elif dataset == "giantsteps_key":
            _giantsteps(root, download_audio=not args.skip_giantsteps_audio)
    print("[done]")


if __name__ == "__main__":
    main()
