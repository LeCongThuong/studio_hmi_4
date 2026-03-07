"""Shared mesh writers and lightweight debug visualization helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def write_ply(path: Path, verts: np.ndarray, faces: Optional[np.ndarray] = None) -> None:
    """Write an ASCII PLY file from vertices and optional triangle indices."""

    verts_arr = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    faces_arr = None if faces is None else np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {verts_arr.shape[0]}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        if faces_arr is not None:
            fp.write(f"element face {faces_arr.shape[0]}\n")
            fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for row in verts_arr:
            fp.write(f"{row[0]} {row[1]} {row[2]}\n")
        if faces_arr is not None:
            for tri in faces_arr:
                fp.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def save_debug_mesh_png(path: Path, verts: np.ndarray, title: str) -> bool:
    """Save a simple 3D scatter preview of a mesh or point cloud."""

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return False

    pts = np.asarray(verts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return False
    step = max(1, int(pts.shape[0] // 4000))
    sample = pts[::step]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
