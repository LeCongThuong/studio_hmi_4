"""Shared `.npy` dictionary I/O helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_npy_dict(path: Path) -> Dict[str, Any]:
    """Load a pickled dict-style `.npy` payload used throughout the pipeline."""

    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and hasattr(obj, "item"):
        data = obj.item()
        if isinstance(data, dict):
            return dict(data)
    if isinstance(obj, dict):
        return dict(obj)
    raise ValueError(f"Unsupported npy dict payload at {path}: {type(obj)}")


def save_npy_dict(path: Path, data: Dict[str, Any]) -> None:
    """Persist a dictionary to `.npy` using the repo's legacy format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, dict(data), allow_pickle=True)
