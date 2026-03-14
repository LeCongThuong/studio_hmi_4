"""I/O and debug helpers for stage-2 triangulation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from studio_hmi_4.common.cv2_compat import require_cv2


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
NP_EXTS = [".npy", ".npz"]


def import_py_module(py_path: str):
    py_path = str(Path(py_path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("mhr_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_existing_with_exts(dir_path: Path, stem: str, exts: Sequence[str]) -> Optional[Path]:
    exts_lower = [ext.lower() for ext in exts]
    for ext in exts:
        candidate = dir_path / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for candidate in dir_path.glob(stem + ".*"):
        if candidate.suffix.lower() in exts_lower:
            return candidate
    return None


def load_pred_keypoints_2d(file_path: Path, index: int) -> np.ndarray:
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        obj = np.load(str(file_path), allow_pickle=True)
        if isinstance(obj, np.ndarray):
            if obj.dtype == object and obj.shape == () and hasattr(obj, "item"):
                data = obj.item()
            else:
                data = {"pred_keypoints_2d": obj}
        elif isinstance(obj, dict):
            data = obj
        else:
            raise ValueError(f"Unsupported .npy payload type at {file_path}: {type(obj)}")
    elif suffix == ".npz":
        archive = np.load(str(file_path), allow_pickle=True)
        if "pred_keypoints_2d" in archive:
            data = {"pred_keypoints_2d": archive["pred_keypoints_2d"]}
        elif "arr_0" in archive:
            arr0 = archive["arr_0"]
            data = arr0.item() if hasattr(arr0, "item") else {"pred_keypoints_2d": arr0}
        else:
            raise KeyError(f"{file_path} has keys {list(archive.keys())}, expected 'pred_keypoints_2d'")
    else:
        raise ValueError(f"Unsupported file: {file_path}")

    if "pred_keypoints_2d" not in data:
        raise KeyError(f"{file_path} missing 'pred_keypoints_2d'. Available keys: {list(data.keys())}")

    arr = np.asarray(data["pred_keypoints_2d"])
    if arr.ndim == 2:
        if arr.shape != (70, 2):
            raise ValueError(f"{file_path}: expected (70,2), got {arr.shape}")
        return arr.astype(np.float64)
    if arr.ndim == 3:
        if arr.shape[1:] != (70, 2):
            raise ValueError(f"{file_path}: expected (N,70,2), got {arr.shape}")
        index = int(np.clip(index, 0, arr.shape[0] - 1))
        return arr[index].astype(np.float64)
    raise ValueError(f"{file_path}: expected ndim 2 or 3, got shape {arr.shape}")


def maybe_denormalize(
    kpts_xy: np.ndarray,
    w: int,
    h: int,
    force_normalized: bool,
    force_pixel: bool,
) -> np.ndarray:
    if force_pixel:
        return kpts_xy.astype(np.float64)

    keypoints = kpts_xy.astype(np.float64).copy()
    finite = np.isfinite(keypoints).all(axis=1)
    if not finite.any():
        return keypoints

    finite_pts = keypoints[finite]
    looks_norm = (
        (finite_pts[:, 0].min() >= -0.5 and finite_pts[:, 0].max() <= 1.5)
        and (finite_pts[:, 1].min() >= -0.5 and finite_pts[:, 1].max() <= 1.5)
    )
    if force_normalized or looks_norm:
        keypoints[:, 0] *= float(w)
        keypoints[:, 1] *= float(h)
    return keypoints


def invalidate_points_outside_image(
    kpts_xy: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    keypoints = np.asarray(kpts_xy, dtype=np.float64).copy()
    finite = np.isfinite(keypoints).all(axis=1)
    if not finite.any():
        return keypoints

    inside = (
        (keypoints[:, 0] >= 0.0)
        & (keypoints[:, 0] < float(w))
        & (keypoints[:, 1] >= 0.0)
        & (keypoints[:, 1] < float(h))
    )
    keypoints[finite & ~inside] = np.nan
    return keypoints


def read_image(img_dir: Path, cam: str, fallback_size_wh: Tuple[int, int]) -> np.ndarray:
    cv2 = require_cv2("stage-2 image loading")
    path = find_existing_with_exts(img_dir, cam, IMG_EXTS)
    if path is None:
        w, h = fallback_size_wh
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        w, h = fallback_size_wh
        return np.zeros((h, w, 3), dtype=np.uint8)
    return img


def show_3d_scatter_interactive(points3d: np.ndarray, title: str) -> None:
    try:
        import matplotlib

        if matplotlib.get_backend().lower() in ("agg", "cairo", "pdf", "svg", "ps"):
            for backend in ("TkAgg", "Qt5Agg", "QtAgg"):
                try:
                    matplotlib.use(backend, force=True)
                    break
                except Exception:
                    pass
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib interactive show failed to init: {exc}")
        return

    points = points3d
    ok = np.isfinite(points).all(axis=1)
    points = points[ok]
    if points.shape[0] == 0:
        print("[DEBUG] No finite 3D points to show.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=14)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def draw_overlay(
    img_bgr: np.ndarray,
    obs: np.ndarray,
    proj: np.ndarray,
    edges: List[Tuple[int, int]],
) -> np.ndarray:
    cv2 = require_cv2("stage-2 overlay drawing")
    out = img_bgr.copy()

    def valid_2d(point: np.ndarray) -> bool:
        return bool(np.isfinite(point).all())

    for a, b in edges:
        if valid_2d(obs[a]) and valid_2d(obs[b]):
            cv2.line(out, tuple(obs[a].astype(int)), tuple(obs[b].astype(int)), (0, 0, 180), 1)
        if valid_2d(proj[a]) and valid_2d(proj[b]):
            cv2.line(out, tuple(proj[a].astype(int)), tuple(proj[b].astype(int)), (0, 180, 0), 1)

    for idx in range(obs.shape[0]):
        if valid_2d(obs[idx]):
            cv2.circle(out, tuple(obs[idx].astype(int)), 3, (0, 0, 255), -1)
        if valid_2d(proj[idx]):
            cv2.circle(out, tuple(proj[idx].astype(int)), 3, (0, 255, 0), -1)
        if valid_2d(obs[idx]) and valid_2d(proj[idx]):
            cv2.line(out, tuple(obs[idx].astype(int)), tuple(proj[idx].astype(int)), (255, 255, 255), 1)

    return out
