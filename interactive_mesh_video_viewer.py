#!/usr/bin/env python3
"""Interactive 4D mesh viewer synchronized with a front-view video.

Reads frame outputs from pipeline optimization folders:
  optimization/<k>/opt_out_smoothed.npy (preferred)
  optimization/<k>/opt_out.npy (fallback)

Then displays:
  1) Interactive Open3D mesh/point-cloud playback (mouse rotate/zoom/pan)
  2) Optional front video window synchronized by frame index

Controls (Open3D window):
  SPACE: play/pause
  N: next frame
  B: previous frame
  R: restart
  L: toggle loop
  + / -: speed up / slow down
  Q: quit
"""
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "open3d is required for interactive mesh visualization. "
        "Install with: pip install open3d"
    ) from exc


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


@dataclass
class FrameMeshRecord:
    rel_dir: str
    frame_index: Optional[int]
    npy_path: Path
    data: Dict[str, Any]


def _natural_tokens(text: str):
    parts = re.split(r"(\d+)", text.replace("\\", "/"))
    out = []
    for p in parts:
        if p.isdigit():
            out.append((0, int(p)))
        else:
            out.append((1, p.lower()))
    return out


def _parse_leaf_index(rel_dir: str) -> Optional[int]:
    leaf = rel_dir.replace("\\", "/").strip("/").split("/")[-1] if rel_dir else ""
    return int(leaf) if leaf.isdigit() else None


def _sort_key(rel_dir: str):
    idx = _parse_leaf_index(rel_dir)
    if idx is not None:
        return (0, idx, rel_dir)
    return (1, _natural_tokens(rel_dir), rel_dir)


def load_npy_dict(path: Path) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and hasattr(obj, "item"):
        data = obj.item()
        if isinstance(data, dict):
            return dict(data)
    if isinstance(obj, dict):
        return dict(obj)
    raise ValueError(f"Unsupported npy payload at {path}: {type(obj)}")


def discover_sequence(
    optimization_root: Path,
    preferred_name: str,
    fallback_name: str,
) -> List[FrameMeshRecord]:
    optimization_root = optimization_root.expanduser().resolve()
    if not optimization_root.exists():
        raise FileNotFoundError(f"Optimization root not found: {optimization_root}")

    pref_map: Dict[str, Path] = {}
    for p in optimization_root.rglob(preferred_name):
        rel = p.parent.relative_to(optimization_root).as_posix()
        pref_map[rel] = p

    fall_map: Dict[str, Path] = {}
    for p in optimization_root.rglob(fallback_name):
        rel = p.parent.relative_to(optimization_root).as_posix()
        fall_map[rel] = p

    keys = sorted(set(pref_map.keys()) | set(fall_map.keys()), key=_sort_key)
    records: List[FrameMeshRecord] = []
    for rel in keys:
        npy_path = pref_map.get(rel, fall_map.get(rel))
        if npy_path is None:
            continue
        try:
            d = load_npy_dict(npy_path)
        except Exception:
            continue
        records.append(
            FrameMeshRecord(
                rel_dir=rel,
                frame_index=_parse_leaf_index(rel),
                npy_path=npy_path,
                data=d,
            )
        )
    return records


def get_vertices(data: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("pred_vertices", "pred_keypoints_3d", "pred_joint_coords"):
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    return None


def get_faces(data: Dict[str, Any]) -> Optional[np.ndarray]:
    faces = data.get("faces", None)
    if faces is None:
        return None
    arr = np.asarray(faces)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)
    return arr


def _resize_for_display(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 0.999:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _find_front_image(frames_root: Path, rel_dir: str, cam: str) -> Optional[Path]:
    rel = Path(rel_dir) if rel_dir else Path(".")
    base = (frames_root / rel).resolve()
    if not base.is_dir():
        return None
    for ext in IMAGE_EXTS:
        p = base / f"{cam}{ext}"
        if p.exists():
            return p
    return None


class FrontProvider:
    def __init__(
        self,
        video_path: Optional[Path],
        frames_root: Optional[Path],
        front_cam: str,
        video_start_idx: int,
        video_stride: int,
        max_w: int,
        max_h: int,
    ):
        self.video_path = None if video_path is None else video_path.expanduser().resolve()
        self.frames_root = None if frames_root is None else frames_root.expanduser().resolve()
        self.front_cam = front_cam
        self.video_start_idx = int(video_start_idx)
        self.video_stride = max(1, int(video_stride))
        self.max_w = max_w
        self.max_h = max_h
        self.cap = None
        self.video_ready = False
        if self.video_path is not None:
            self.cap = cv2.VideoCapture(str(self.video_path))
            self.video_ready = bool(self.cap.isOpened())

    def close(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyWindow("Front View")

    def _read_video_frame(self, seq_idx: int) -> Optional[np.ndarray]:
        if not self.video_ready or self.cap is None:
            return None
        v_idx = self.video_start_idx + seq_idx * self.video_stride
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(v_idx)))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _read_front_image(self, rel_dir: str) -> Optional[np.ndarray]:
        if self.frames_root is None:
            return None
        p = _find_front_image(self.frames_root, rel_dir, self.front_cam)
        if p is None:
            return None
        frame = cv2.imread(str(p))
        return frame

    def show(
        self,
        seq_idx: int,
        rel_dir: str,
        total: int,
        is_recovered: bool,
        is_smoothed: bool,
    ):
        frame = self._read_video_frame(seq_idx)
        if frame is None:
            frame = self._read_front_image(rel_dir)
        if frame is None:
            frame = np.zeros((420, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "No front view source",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
        frame = _resize_for_display(frame, max_w=self.max_w, max_h=self.max_h)
        tag = []
        if is_recovered:
            tag.append("recovered")
        if is_smoothed:
            tag.append("smoothed")
        suffix = "" if not tag else (" | " + ",".join(tag))
        text = f"frame {seq_idx + 1}/{total} | rel={rel_dir}{suffix}"
        cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Front View", frame)
        cv2.waitKey(1)


def _print_controls():
    print("Controls (Open3D):")
    print("  SPACE: play/pause")
    print("  N: next frame")
    print("  B: previous frame")
    print("  R: restart")
    print("  L: toggle loop")
    print("  + / -: speed up / slow down")
    print("  Q: quit")
    print("  Mouse: rotate / pan / zoom")


def run_viewer(args: argparse.Namespace) -> None:
    records = discover_sequence(
        optimization_root=Path(args.optimization_root),
        preferred_name=args.npy_name,
        fallback_name=args.fallback_npy_name,
    )
    if not records:
        raise FileNotFoundError(
            f"No sequence npy found under {args.optimization_root} "
            f"(expected {args.npy_name} or {args.fallback_npy_name})."
        )

    vertices0 = get_vertices(records[0].data)
    if vertices0 is None:
        raise KeyError(f"No mesh/point keys found in first frame: {records[0].npy_path}")
    faces = get_faces(records[0].data)
    use_mesh = faces is not None and args.force_points is False

    front = FrontProvider(
        video_path=None if args.front_video is None else Path(args.front_video),
        frames_root=None if args.input_frames_root is None else Path(args.input_frames_root),
        front_cam=args.front_cam,
        video_start_idx=args.video_start_idx,
        video_stride=args.video_stride,
        max_w=args.front_max_w,
        max_h=args.front_max_h,
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Mesh Reconstruction (Interactive)",
        width=args.window_width,
        height=args.window_height,
    )
    if use_mesh:
        geom = o3d.geometry.TriangleMesh()
        geom.vertices = o3d.utility.Vector3dVector(vertices0.astype(np.float64))
        geom.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        geom.compute_vertex_normals()
        vis.add_geometry(geom)
    else:
        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(vertices0.astype(np.float64))
        vis.add_geometry(geom)
        vis.get_render_option().point_size = float(args.point_size)

    state = {
        "idx": max(0, min(int(args.start_index), len(records) - 1)),
        "playing": True,
        "loop": (not args.no_loop),
        "fps": float(args.fps),
        "quit": False,
        "last": time.time(),
    }

    def _apply_frame(i: int, force_normals: bool = False) -> None:
        rec = records[i]
        verts = get_vertices(rec.data)
        if verts is None:
            return
        if use_mesh:
            geom.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
            if force_normals:
                geom.compute_vertex_normals()
        else:
            geom.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
        vis.update_geometry(geom)
        front.show(
            seq_idx=i,
            rel_dir=rec.rel_dir,
            total=len(records),
            is_recovered=bool(int(np.asarray(rec.data.get("opt_recovered", 0)).reshape(()))),
            is_smoothed=bool(int(np.asarray(rec.data.get("opt_smoothed", 0)).reshape(()))),
        )

    def _step(delta: int):
        if len(records) == 0:
            return
        nxt = state["idx"] + int(delta)
        if state["loop"]:
            nxt %= len(records)
        else:
            nxt = max(0, min(nxt, len(records) - 1))
        state["idx"] = nxt
        _apply_frame(
            state["idx"],
            force_normals=use_mesh and (state["idx"] % max(1, int(args.recompute_normals_every)) == 0),
        )

    def _toggle_play(_vis):
        state["playing"] = not state["playing"]
        return False

    def _next(_vis):
        state["playing"] = False
        _step(1)
        return False

    def _prev(_vis):
        state["playing"] = False
        _step(-1)
        return False

    def _restart(_vis):
        state["idx"] = 0
        _apply_frame(state["idx"], force_normals=use_mesh)
        return False

    def _toggle_loop(_vis):
        state["loop"] = not state["loop"]
        return False

    def _faster(_vis):
        state["fps"] = min(120.0, state["fps"] * 1.25)
        return False

    def _slower(_vis):
        state["fps"] = max(0.25, state["fps"] / 1.25)
        return False

    def _quit(_vis):
        state["quit"] = True
        return False

    vis.register_key_callback(ord(" "), _toggle_play)
    vis.register_key_callback(ord("N"), _next)
    vis.register_key_callback(ord("B"), _prev)
    vis.register_key_callback(ord("R"), _restart)
    vis.register_key_callback(ord("L"), _toggle_loop)
    vis.register_key_callback(ord("="), _faster)  # often same key as '+'
    vis.register_key_callback(ord("+"), _faster)
    vis.register_key_callback(ord("-"), _slower)
    vis.register_key_callback(ord("Q"), _quit)

    _print_controls()
    _apply_frame(state["idx"], force_normals=use_mesh)

    try:
        while not state["quit"]:
            if not vis.poll_events():
                break
            now = time.time()
            if state["playing"] and (now - state["last"]) >= (1.0 / max(1e-4, state["fps"])):
                _step(1)
                state["last"] = now
            vis.update_renderer()
            time.sleep(0.001)
    finally:
        front.close()
        vis.destroy_window()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Interactive mesh viewer synchronized with front video for pipeline outputs."
    )
    ap.add_argument("--optimization_root", required=True, type=str, help="Path to <output_root>/optimization")
    ap.add_argument("--npy_name", type=str, default="opt_out_smoothed.npy", help="Preferred per-frame npy name.")
    ap.add_argument("--fallback_npy_name", type=str, default="opt_out.npy", help="Fallback per-frame npy name.")
    ap.add_argument("--front_video", type=str, default=None, help="Optional front camera video path (mp4/avi).")
    ap.add_argument("--input_frames_root", type=str, default=None, help="Optional input frame tree root for fallback image display.")
    ap.add_argument("--front_cam", type=str, default="front", help="Front camera stem for fallback image mode.")
    ap.add_argument("--video_start_idx", type=int, default=0, help="Start frame index in front video aligned with sequence index 0.")
    ap.add_argument("--video_stride", type=int, default=1, help="Video frame stride per sequence frame.")
    ap.add_argument("--fps", type=float, default=20.0, help="Playback fps for mesh sequence.")
    ap.add_argument("--start_index", type=int, default=0, help="Initial sequence frame index.")
    ap.add_argument("--no_loop", action="store_true", help="Disable loop playback.")
    ap.add_argument("--force_points", action="store_true", help="Render as point cloud even if faces are available.")
    ap.add_argument("--point_size", type=float, default=4.0, help="Point size in point-cloud mode.")
    ap.add_argument("--recompute_normals_every", type=int, default=2, help="Recompute mesh normals every N frames.")
    ap.add_argument("--window_width", type=int, default=1280, help="Open3D window width.")
    ap.add_argument("--window_height", type=int, default=860, help="Open3D window height.")
    ap.add_argument("--front_max_w", type=int, default=960, help="Max width of front video window.")
    ap.add_argument("--front_max_h", type=int, default=720, help="Max height of front video window.")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_viewer(args)


if __name__ == "__main__":
    main()
