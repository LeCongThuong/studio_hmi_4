"""Shared discovery helpers for per-frame pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .sorting import parse_leaf_index, sort_key


@dataclass
class FrameFileRecord:
    """A discovered file associated with a relative frame directory."""

    rel_dir: str
    frame_index: Optional[int]
    path: Path


def discover_preferred_frame_files(
    root: Path,
    preferred_name: str,
    fallback_name: str,
) -> List[FrameFileRecord]:
    """Discover one preferred file per relative frame directory.

    If both preferred and fallback files exist in the same frame directory, the
    preferred file wins.
    """

    root = root.expanduser().resolve()

    preferred: dict[str, Path] = {}
    for path in root.rglob(preferred_name):
        rel_dir = path.parent.relative_to(root).as_posix()
        preferred[rel_dir] = path

    fallback: dict[str, Path] = {}
    for path in root.rglob(fallback_name):
        rel_dir = path.parent.relative_to(root).as_posix()
        fallback[rel_dir] = path

    records: List[FrameFileRecord] = []
    for rel_dir in sorted(set(preferred) | set(fallback), key=sort_key):
        selected = preferred.get(rel_dir, fallback.get(rel_dir))
        if selected is None:
            continue
        records.append(
            FrameFileRecord(
                rel_dir=rel_dir,
                frame_index=parse_leaf_index(rel_dir),
                path=selected,
            )
        )
    return records
