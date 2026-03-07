"""I/O and persistence helpers for stage-1 SAM-3D inference."""

from __future__ import annotations

import os
import re
import shutil
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from studio_hmi_4.common import (
    normalize_rel_dir as _normalize_rel_dir,
    relative_dir as _relative_dir,
    save_npy_dict,
)


IMAGE_EXTENSIONS: Sequence[str] = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.gif",
    "*.bmp",
    "*.tiff",
    "*.webp",
)

MHR_PARAM_KEYS: Sequence[str] = (
    "body_pose_params",
    "hand_pose_params",
    "scale_params",
    "shape_params",
    "expr_params",
)


def resolve_output_folder(image_folder: str, output_folder: str) -> Path:
    if output_folder:
        return Path(output_folder)
    return Path("./output") / Path(image_folder).name


def ensure_output_dirs(
    output_root: Path,
    debug: bool,
    save_mhr_params: bool,
) -> tuple[Path, Path, Path, Optional[Path]]:
    render_root = output_root / "render"
    npy_root = output_root / "npy"
    mesh_root = output_root / "mesh"
    mhr_root = output_root / "mhr_params" if save_mhr_params else None

    npy_root.mkdir(parents=True, exist_ok=True)
    if debug:
        render_root.mkdir(parents=True, exist_ok=True)
        mesh_root.mkdir(parents=True, exist_ok=True)
    if mhr_root is not None:
        mhr_root.mkdir(parents=True, exist_ok=True)

    return render_root, npy_root, mesh_root, mhr_root


def resolve_model_paths(config: object) -> tuple[str, str, str, str]:
    mhr_path = getattr(config, "mhr_path", "") or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = getattr(config, "detector_path", "") or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = getattr(config, "segmentor_path", "") or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = getattr(config, "fov_path", "") or os.environ.get("SAM3D_FOV_PATH", "")
    return mhr_path, detector_path, segmentor_path, fov_path


def collect_images(image_root: str) -> List[str]:
    def _natural_key(path: str):
        rel = os.path.relpath(path, image_root)
        parts = re.split(r"(\d+)", rel.replace("\\", "/"))
        out = []
        for token in parts:
            if token.isdigit():
                out.append((0, f"{int(token):020d}"))
            else:
                out.append((1, token.lower()))
        return out

    images = [
        image
        for ext in IMAGE_EXTENSIONS
        for image in glob(os.path.join(image_root, "**", ext), recursive=True)
    ]
    return sorted(images, key=_natural_key)


def collect_ply_paths(ply_files: Optional[Iterable[str]]) -> List[str]:
    if ply_files is None:
        return []

    if isinstance(ply_files, (list, tuple)):
        candidates = [str(path) for path in ply_files if path is not None]
    else:
        candidates = [str(ply_files)]

    return [path for path in candidates if os.path.isfile(path) and path.lower().endswith(".ply")]


def move_mesh_files(
    ply_files: Optional[Iterable[str]],
    mesh_out_dir: Path,
    image_name: str,
) -> None:
    ply_paths = collect_ply_paths(ply_files)
    if not ply_paths:
        return

    if len(ply_paths) == 1:
        shutil.move(ply_paths[0], mesh_out_dir / f"{image_name}.ply")
        return

    for src in ply_paths:
        shutil.move(src, mesh_out_dir / Path(src).name)


def extract_primary_output(
    outputs: object,
    strategy: str = "largest_bbox",
    person_index: int = 0,
) -> Optional[Dict[str, object]]:
    if outputs is None:
        return None

    if isinstance(outputs, Mapping):
        return dict(outputs)

    if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
        return None

    candidates = [item for item in outputs if isinstance(item, Mapping)]
    if len(candidates) == 0:
        return None

    if strategy == "first":
        return dict(candidates[0])
    if strategy == "person_index":
        idx = int(np.clip(int(person_index), 0, len(candidates) - 1))
        return dict(candidates[idx])

    def area(candidate: Mapping[str, object]) -> float:
        bbox_obj = candidate.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox = np.asarray(bbox_obj, dtype=np.float32).reshape(-1)
        if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
            return -1.0
        x0, y0, x1, y1 = bbox[:4]
        return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))

    best_candidate = candidates[0]
    best_area = area(best_candidate)
    for candidate in candidates[1:]:
        candidate_area = area(candidate)
        if candidate_area > best_area:
            best_candidate = candidate
            best_area = candidate_area
    return dict(best_candidate)


def save_dict_npy(data: Mapping[str, object], output_folder: Path, image_name: str) -> Path:
    out_path = output_folder / f"{image_name}.npy"
    save_npy_dict(out_path, dict(data))
    return out_path


def extract_mhr_params(
    output_dict: Mapping[str, object],
    strict: bool = False,
    keys: Sequence[str] = MHR_PARAM_KEYS,
) -> Dict[str, np.ndarray]:
    mhr_params: Dict[str, np.ndarray] = {}
    missing: List[str] = []

    for key in keys:
        if key not in output_dict:
            missing.append(key)
            continue
        mhr_params[key] = np.asarray(output_dict[key])

    if strict and missing:
        raise KeyError(f"Missing expected MHR keys: {missing}")
    return mhr_params


def filter_images_by_rel_dirs(
    images: List[str],
    image_root: str,
    rel_dirs: Sequence[str],
) -> List[str]:
    allowed = {_normalize_rel_dir(rel_dir) for rel_dir in rel_dirs}
    filtered: List[str] = []
    for image_path in images:
        rel_dir = _normalize_rel_dir(_relative_dir(image_path, image_root))
        if rel_dir in allowed:
            filtered.append(image_path)
    return filtered
