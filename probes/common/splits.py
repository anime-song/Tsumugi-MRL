"""Deterministic split helpers for datasets without official splits."""

from __future__ import annotations

import hashlib


def hash_split(key: str, *, train_fraction: float = 0.8, valid_fraction: float = 0.1) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_fraction:
        return "train"
    if value < train_fraction + valid_fraction:
        return "valid"
    return "test"


def group_splits(groups: list[str]) -> dict[str, str]:
    """Assign complete groups to splits in a stable, approximately balanced way."""

    unique = sorted(set(groups))
    assignments = {group: hash_split(group) for group in unique}
    if "train" not in assignments.values() or "test" not in assignments.values():
        for index, group in enumerate(unique):
            assignments[group] = ["train", "valid", "test"][index % 3]
    return assignments
