"""Shared helpers for compact export and official-MHR forward."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from studio_hmi_4.common import parse_leaf_index, sort_key
from studio_hmi_4.common.frame_files import discover_preferred_frame_files


@dataclass
class ExportFrameRecord:
    rel_dir: str
    frame_index: Optional[int]
    path: Path


def discover_frame_exports(
    root: Path,
    preferred_name: str,
    fallback_name: str,
) -> List[ExportFrameRecord]:
    records = discover_preferred_frame_files(
        root=root,
        preferred_name=preferred_name,
        fallback_name=fallback_name,
    )
    return [
        ExportFrameRecord(
            rel_dir=record.rel_dir,
            frame_index=record.frame_index,
            path=record.path,
        )
        for record in records
    ]


def discover_named_exports(root: Path, file_name: str) -> List[ExportFrameRecord]:
    found = [
        ExportFrameRecord(
            rel_dir=path.parent.relative_to(root).as_posix(),
            frame_index=parse_leaf_index(path.parent.relative_to(root).as_posix()),
            path=path,
        )
        for path in root.rglob(file_name)
    ]
    found.sort(key=lambda record: sort_key(record.rel_dir))
    return found


def load_params_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: np.asarray(data[key]) for key in data.files}


def extract_vec(
    data: Dict[str, Any],
    key: str,
    dim: int,
) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {list(data.keys())}")
    value = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if value.size != dim:
        raise ValueError(f"Key '{key}' has dim {value.size}, expected {dim}")
    if not np.isfinite(value).all():
        raise ValueError(f"Key '{key}' contains non-finite values")
    return value


def extract_optional_scalar_int(
    data: Dict[str, Any],
    key: str,
    default: int = 0,
) -> int:
    if key not in data:
        return int(default)
    return int(np.asarray(data[key]).reshape(()))


def extract_optional_vec(
    data: Dict[str, Any],
    key: str,
    expected_dim: Optional[int] = None,
) -> Optional[np.ndarray]:
    if key not in data:
        return None
    value = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if expected_dim is not None and value.size != expected_dim:
        raise ValueError(f"{key} has dim {value.size}, expected {expected_dim}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value
