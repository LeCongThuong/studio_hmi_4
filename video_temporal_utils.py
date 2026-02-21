#!/usr/bin/env python3
"""Utilities for sequence-level recovery, smoothing, and debug playback."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_SMOOTH_KEYS: Sequence[str] = (
    "body_pose_params",
    "pred_keypoints_3d",
    "pred_joint_coords",
)
DEFAULT_MEDIAN_WINDOW = 5
DEFAULT_OUTLIER_SIGMA = 3.5


def load_npy_dict(path: Path) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and hasattr(obj, "item"):
        data = obj.item()
        if isinstance(data, dict):
            return dict(data)
    if isinstance(obj, dict):
        return dict(obj)
    raise ValueError(f"Unsupported npy dict payload at {path}: {type(obj)}")


def save_npy_dict(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, dict(data), allow_pickle=True)


def _is_numeric_compatible(a: Any, b: Any) -> bool:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.shape != arr_b.shape:
        return False
    if arr_a.dtype == object or arr_b.dtype == object:
        return False
    return bool(np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number))


def copy_frame_dict(src: Dict[str, Any], mode: str) -> Dict[str, Any]:
    out = dict(src)
    out["opt_recovered"] = int(1)
    out["opt_recovery_mode"] = mode
    return out


def interpolate_frame_dict(
    prev_dict: Dict[str, Any],
    next_dict: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """Interpolate numeric keys that are shape-compatible; copy the rest."""
    a = float(np.clip(alpha, 0.0, 1.0))
    base = dict(prev_dict if a < 0.5 else next_dict)
    for key in list(base.keys()):
        if key not in prev_dict or key not in next_dict:
            continue
        if not _is_numeric_compatible(prev_dict[key], next_dict[key]):
            continue
        p = np.asarray(prev_dict[key], dtype=np.float32)
        n = np.asarray(next_dict[key], dtype=np.float32)
        base[key] = ((1.0 - a) * p + a * n).astype(np.float32)

    base["opt_recovered"] = int(1)
    base["opt_recovery_mode"] = "interpolate"
    base["opt_recovery_alpha"] = float(a)
    return base


def _nearest_fill(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = values.copy()
    idx = np.where(valid_mask)[0]
    if idx.size == 0:
        return out
    for t in range(values.shape[0]):
        if valid_mask[t]:
            continue
        nearest = int(idx[np.argmin(np.abs(idx - t))])
        out[t] = values[nearest]
    return out


def _bidirectional_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 1e-4, 1.0))
    fw = values.copy()
    bw = values.copy()
    for i in range(1, values.shape[0]):
        fw[i] = a * values[i] + (1.0 - a) * fw[i - 1]
    for i in range(values.shape[0] - 2, -1, -1):
        bw[i] = a * values[i] + (1.0 - a) * bw[i + 1]
    return 0.5 * (fw + bw)


def _temporal_median(values: np.ndarray, window: int) -> np.ndarray:
    if values.shape[0] <= 1:
        return values.copy()
    w = max(1, int(window))
    if w <= 1:
        return values.copy()
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.empty_like(values)
    for t in range(values.shape[0]):
        lo = max(0, t - half)
        hi = min(values.shape[0], t + half + 1)
        out[t] = np.median(values[lo:hi], axis=0)
    return out


def _suppress_outliers(values: np.ndarray, median_seq: np.ndarray, sigma: float) -> np.ndarray:
    s = float(sigma)
    if s <= 0:
        return values
    diff = values - median_seq
    mad = np.median(np.abs(diff), axis=0) + 1e-6
    robust_std = 1.4826 * mad
    thresh = s * robust_std
    return np.where(np.abs(diff) > thresh, median_seq, values)


def smooth_frame_dict_sequence(
    frame_dicts: Sequence[Optional[Dict[str, Any]]],
    alpha: float = 0.65,
    keys: Sequence[str] = DEFAULT_SMOOTH_KEYS,
    median_window: int = DEFAULT_MEDIAN_WINDOW,
    outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
) -> List[Optional[Dict[str, Any]]]:
    """Return new dicts with temporally smoothed numeric fields."""
    out: List[Optional[Dict[str, Any]]] = [None if d is None else dict(d) for d in frame_dicts]
    if len(out) == 0:
        return out

    for key in keys:
        sample_shape = None
        for d in out:
            if d is None or key not in d:
                continue
            arr = np.asarray(d[key])
            if arr.dtype == object:
                continue
            sample_shape = arr.shape
            break
        if sample_shape is None:
            continue

        seq = np.zeros((len(out),) + sample_shape, dtype=np.float32)
        valid = np.zeros((len(out),), dtype=bool)
        for i, d in enumerate(out):
            if d is None or key not in d:
                continue
            arr = np.asarray(d[key], dtype=np.float32)
            if arr.shape != sample_shape or arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
                continue
            if not np.isfinite(arr).all():
                continue
            seq[i] = arr.astype(np.float32)
            valid[i] = True
        if int(valid.sum()) < 2:
            continue

        filled = _nearest_fill(seq, valid)
        median_seq = _temporal_median(filled, window=median_window)
        filtered = _suppress_outliers(filled, median_seq, sigma=outlier_sigma)
        smoothed = _bidirectional_ema(filtered, alpha=alpha)
        for i, d in enumerate(out):
            if d is None:
                continue
            if not np.isfinite(smoothed[i]).all():
                continue
            d[key] = smoothed[i].astype(np.float32)

    for d in out:
        if d is None:
            continue
        d["opt_smoothed"] = int(1)
        d["opt_smoothing_alpha"] = float(alpha)
        d["opt_smoothing_median_window"] = int(max(1, int(median_window)))
        d["opt_smoothing_outlier_sigma"] = float(outlier_sigma)
    return out


def extract_keypoint_sequence(
    frame_dicts: Sequence[Optional[Dict[str, Any]]],
    key: str = "pred_keypoints_3d",
) -> np.ndarray:
    """Build (T,N,3) array with NaNs for missing frames."""
    shape = None
    for d in frame_dicts:
        if d is None or key not in d:
            continue
        arr = np.asarray(d[key], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            shape = arr.shape
            break
    if shape is None:
        return np.zeros((0, 0, 3), dtype=np.float32)

    seq = np.full((len(frame_dicts), shape[0], 3), np.nan, dtype=np.float32)
    for i, d in enumerate(frame_dicts):
        if d is None or key not in d:
            continue
        arr = np.asarray(d[key], dtype=np.float32)
        if arr.shape == shape:
            seq[i] = arr
    return seq


def _stable_3d_limits(points_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points_seq).all(axis=2)
    if not valid.any():
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    pts = points_seq[valid]
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    center = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins) * 0.55 + 1e-6)
    lo = center - radius
    hi = center + radius
    return lo, hi


def save_keypoint_sequence_mp4(
    points_seq: np.ndarray,
    out_mp4: Path,
    fps: int = 20,
    title: str = "4D Reconstruction",
) -> bool:
    if points_seq.ndim != 3 or points_seq.shape[-1] != 3 or points_seq.shape[0] == 0:
        return False
    try:
        import cv2
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] MP4 export unavailable: {exc}")
        return False

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = _stable_3d_limits(points_seq)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout()
    fig.canvas.draw()
    h, w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(fps)),
        (w, h),
    )
    if not writer.isOpened():
        plt.close(fig)
        print(f"[WARN] Failed to open VideoWriter for: {out_mp4}")
        return False

    for i in range(points_seq.shape[0]):
        ax.cla()
        pts = points_seq[i]
        ok = np.isfinite(pts).all(axis=1)
        if ok.any():
            p = pts[ok]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=12, c="tab:blue")
        ax.set_title(f"{title} | frame={i:04d}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_zlim(float(lo[2]), float(hi[2]))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)
    return True


def show_keypoint_sequence_interactive(
    points_seq: np.ndarray,
    title: str = "4D Reconstruction",
    interval_ms: int = 60,
) -> bool:
    if points_seq.ndim != 3 or points_seq.shape[-1] != 3 or points_seq.shape[0] == 0:
        return False
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:
        print(f"[WARN] Interactive debug unavailable: {exc}")
        return False

    lo, hi = _stable_3d_limits(points_seq)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _update(i: int):
        ax.cla()
        pts = points_seq[i]
        ok = np.isfinite(pts).all(axis=1)
        if ok.any():
            p = pts[ok]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=15, c="tab:orange")
        ax.set_title(f"{title} | frame={i:04d}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_zlim(float(lo[2]), float(hi[2]))
        return []

    _anim = FuncAnimation(  # noqa: F841
        fig,
        _update,
        frames=points_seq.shape[0],
        interval=max(1, int(interval_ms)),
        repeat=True,
    )
    plt.show()
    return True
