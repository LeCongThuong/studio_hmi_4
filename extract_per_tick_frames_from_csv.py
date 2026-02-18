#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch extract per-tick frames from multiple camera videos using PyAV,
driven by the *Recorder output* (meta/frame_times.csv + meta/session_meta.json).

This script is specifically adapted for the CSV/JSON produced by the
"Multi-Camera Recorder" code you shared (the one that writes one frame per tick).

Why this is better than timestamp-matching
------------------------------------------
Your recorder writes exactly ONE frame per camera per tick k, in order:
  - for every camera video: output frame index == k

So extraction can be exact and simple:
  - to get tick k for camera 'front', decode frame index k of videos/front.mp4

Expected session folder structure
---------------------------------
root_dir/
  meta/
    frame_times.csv
    session_meta.json        (recommended, used to map slug -> exact video file)
  videos/
    <slug>.mp4               (or .avi fallback)

Expected CSV format (Recorder output)
-------------------------------------
meta/frame_times.csv has columns like:
  k, tick_mono_ns, tick_video_ms,
    <slug>_out_idx, <slug>_sel_mono_ns, <slug>_sel_dt_ms, <slug>_is_fallback,
    ...

We only NEED:
  - k (tick id)
  - tick_video_ms (optional, used only for naming and the index CSV)
  - <slug>_out_idx (usually equals k; kept for sanity)

Camera selection (--cams)
------------------------
- omit --cams        -> use all cameras that exist as both CSV camera columns and video files
- --cams all         -> same as omit
- --cams front,left  -> restrict to these cameras

Output
------
For each tick id (k), writes:
  OUT/<k>/<cam>.jpg (or png)

Also writes:
  OUT/frames_index.csv

Usage
-----
pip install av pillow numpy
python extract_per_tick_frames_from_recorder_csv.py \
  --root-dir /path/to/session_root \
  --out /path/to/output \
  --cams all

Notes
-----
- If a video is shorter than the CSV (rare but possible), missing frames are skipped
  and reported.
- If the recorder fell back to a black frame for some tick (is_fallback=1),
  that black frame is still inside the video at frame index k, so extraction remains exact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import av
import numpy as np
from PIL import Image

VIDEO_EXTENSIONS = (".mp4", ".avi")


# --------------------- CLI ---------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract per-tick frames from multi-camera videos using Recorder CSV/JSON (PyAV)."
    )
    ap.add_argument("--root-dir", required=True, type=Path,
                    help="Session root dir containing meta/frame_times.csv and videos/")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination directory. Creates OUT/<k>/<cam>.(jpg|png)")
    ap.add_argument("--id-key", default="k", type=str,
                    help="CSV column name for tick id (default: k)")
    ap.add_argument("--video-subdir", default="videos", type=str,
                    help="Subdir under root-dir containing videos (default: videos)")
    ap.add_argument("--meta-subdir", default="meta", type=str,
                    help="Subdir under root-dir containing CSV/JSON (default: meta)")
    ap.add_argument("--frame-times", default="frame_times.csv", type=str,
                    help="CSV filename under meta-subdir (default: frame_times.csv)")
    ap.add_argument("--session-meta", default="session_meta.json", type=str,
                    help="Session meta JSON filename under meta-subdir (default: session_meta.json)")
    ap.add_argument("--cams", default=None, type=str,
                    help=("Camera selection: omit or 'all' = all cameras; "
                          "or comma list like 'front,left'"))
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"],
                    help="Image format (default: jpg)")
    ap.add_argument("--name-with-time", action="store_true",
                    help="If set, filenames become <cam>_t<tick_video_ms>.ext")
    return ap.parse_args()


# --------------------- CSV utils ---------------------

def _sniff_dialect(csv_path: Path) -> csv.Dialect:
    """
    Try to sniff delimiter (comma/tab/semicolon/etc.). Falls back to excel dialect.
    """
    sample = csv_path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except Exception:
        return csv.get_dialect("excel")


def _parse_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        # handle "12.0" if it ever appears
        return int(float(s))
    except Exception:
        return None


def discover_cam_keys_from_csv(frame_times_path: Path, suffix: str = "_out_idx") -> List[str]:
    """
    Discover camera slugs from columns like '<slug>_out_idx'.
    """
    dialect = _sniff_dialect(frame_times_path)
    with frame_times_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            return []
        cams = []
        for col in reader.fieldnames:
            if col.endswith(suffix):
                cams.append(col[: -len(suffix)])
        return cams


def read_targets_from_recorder_csv(
    frame_times_path: Path,
    id_key: str,
    allowed_cam_keys: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, List[Tuple[int, str, Optional[int]]]]]:
    """
    Parse Recorder-style frame_times.csv and build per-camera extraction targets by frame index.

    Returns
    -------
    id_order:
      list of ids (string) in file order (usually "0","1",...).
    targets_by_cam:
      dict: cam_slug -> list of (frame_index, id_str, tick_video_ms)
      - frame_index is taken from '<slug>_out_idx' (usually equals k)
      - tick_video_ms is taken from 'tick_video_ms' if present, else None
    """
    dialect = _sniff_dialect(frame_times_path)
    targets_by_cam: Dict[str, List[Tuple[int, str, Optional[int]]]] = {}
    id_order: List[str] = []

    with frame_times_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV / no header: {frame_times_path}")

        if id_key not in reader.fieldnames:
            raise KeyError(f"Missing id_key '{id_key}' in CSV header: {frame_times_path}")

        # Find which cameras exist via *_out_idx columns
        cam_cols: Dict[str, str] = {}
        for col in reader.fieldnames:
            if col.endswith("_out_idx"):
                cam = col[:-len("_out_idx")]
                if allowed_cam_keys is not None and cam not in allowed_cam_keys:
                    continue
                cam_cols[cam] = col

        if not cam_cols:
            raise RuntimeError(f"No '*_out_idx' columns found in: {frame_times_path}")

        for ln, row in enumerate(reader, start=2):
            kid = str(row.get(id_key, "")).strip()
            if kid == "":
                continue
            id_order.append(kid)

            tick_video_ms = _parse_int(row.get("tick_video_ms"))

            for cam, col in cam_cols.items():
                out_idx = _parse_int(row.get(col))
                if out_idx is None or out_idx < 0:
                    continue
                targets_by_cam.setdefault(cam, []).append((out_idx, kid, tick_video_ms))

    # Ensure targets sorted by frame index (for sequential decode)
    for cam in list(targets_by_cam.keys()):
        targets_by_cam[cam].sort(key=lambda x: x[0])

    return id_order, targets_by_cam


# --------------------- session_meta.json utils ---------------------

def load_session_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load meta/session_meta.json if it exists; return dict or None.
    """
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_video_files(video_dir: Path):
    for ext in VIDEO_EXTENSIONS:
        yield from video_dir.glob(f"*{ext}")


def build_video_map_from_meta(root_dir: Path, meta: Optional[Dict[str, Any]], video_dir: Path) -> Dict[str, Path]:
    """
    Build slug -> video path mapping.

    Priority:
      1) session_meta.json: cameras[].slug + cameras[].video_file (most exact)
      2) fallback: scan videos/ for *.mp4 + *.avi, map stem -> path
    """
    video_map: Dict[str, Path] = {}

    # 1) From session_meta.json
    if meta and isinstance(meta.get("cameras"), list):
        for cam in meta["cameras"]:
            if not isinstance(cam, dict):
                continue
            slug = str(cam.get("slug", "")).strip()
            vfile = str(cam.get("video_file", "")).strip()
            if not slug:
                continue
            if vfile:
                p = (root_dir / vfile).resolve()
                if p.exists():
                    video_map[slug] = p

    # 2) Fallback scan
    if not video_map:
        for p in iter_video_files(video_dir):
            video_map[p.stem] = p.resolve()
    else:
        # Still add any missing slugs from scan (in case meta is incomplete)
        for p in iter_video_files(video_dir):
            video_map.setdefault(p.stem, p.resolve())

    return video_map


def parse_camera_selection(
    cams_arg: Optional[str],
    cams_with_video: Set[str],
) -> Set[str]:
    if cams_arg is None or cams_arg.strip().lower() == "all":
        return cams_with_video

    requested = {c.strip() for c in cams_arg.split(",") if c.strip()}
    missing = sorted(requested - cams_with_video)
    if missing:
        raise RuntimeError(
            "Some requested cameras are not available as BOTH CSV column and video file:\n"
            f"  missing: {missing}\n"
            f"  available: {sorted(cams_with_video)}"
        )
    return requested


# --------------------- image save ---------------------

def save_bgr(path: Path, bgr: np.ndarray) -> None:
    """
    Save BGR uint8 image (OpenCV-style) using PIL.
    """
    rgb = bgr[:, :, ::-1]
    Image.fromarray(rgb).save(str(path))


# --------------------- Extraction core (frame-index based) ---------------------

def extract_for_one_camera_by_index(
    video_path: Path,
    cam_key: str,
    targets: List[Tuple[int, str, Optional[int]]],
    out_root: Path,
    ext: str,
    name_with_time: bool,
) -> List[Tuple[str, str, int, Optional[int], float, int, Path]]:
    """
    Decode a camera video once and save frames at specific frame indices.

    Parameters
    ----------
    video_path:
      The camera video path (mp4 or avi).
    cam_key:
      Camera slug (e.g., 'front').
    targets:
      List of (frame_idx, id_str, tick_video_ms), sorted by frame_idx.
      frame_idx is the output video frame index to extract (usually equals k).
    out_root:
      Output dir.
    ext:
      jpg|png
    name_with_time:
      If True, filename is <cam>_t<tick_video_ms>.ext when tick_video_ms exists,
      otherwise falls back to <cam>_k<id>.ext.

    Returns
    -------
    rows:
      (id, camera, frame_idx, tick_video_ms, actual_time_ms, decoded_frame_idx, path)
    """
    rows: List[Tuple[str, str, int, Optional[int], float, int, Path]] = []
    if not targets:
        return rows

    # Group targets by frame_idx (just in case duplicates occur)
    by_idx: Dict[int, List[Tuple[str, Optional[int]]]] = {}
    for frame_idx, id_str, tick_ms in targets:
        by_idx.setdefault(int(frame_idx), []).append((id_str, tick_ms))
    wanted_indices = sorted(by_idx.keys())
    wanted_set = set(wanted_indices)

    container = av.open(str(video_path))
    vstream = next((s for s in container.streams if s.type == "video"), None)
    if vstream is None:
        container.close()
        raise RuntimeError(f"No video stream in {video_path}")

    time_base = float(vstream.time_base)  # seconds per PTS tick

    # Sequential decode; track decoded frame index
    decoded_idx = -1
    next_wanted_i = 0
    last_wanted = wanted_indices[-1]

    for packet in container.demux(vstream):
        for frame in packet.decode():
            decoded_idx += 1
            if decoded_idx > last_wanted:
                break

            if decoded_idx not in wanted_set:
                continue

            # Convert PTS to ms if present (nice for index CSV)
            if frame.pts is not None:
                actual_time_ms = float(frame.pts) * time_base * 1000.0
            else:
                actual_time_ms = float("nan")

            img = frame.to_ndarray(format="bgr24")

            # Save into each OUT/<id>/ for this index
            for (id_str, tick_ms) in by_idx.get(decoded_idx, []):
                id_dir = out_root / str(id_str)
                id_dir.mkdir(parents=True, exist_ok=True)

                if name_with_time:
                    if tick_ms is not None:
                        fname = f"{cam_key}_t{int(tick_ms):013d}.{ext}"
                    else:
                        fname = f"{cam_key}_k{str(id_str)}.{ext}"
                else:
                    fname = f"{cam_key}.{ext}"

                out_path = id_dir / fname
                save_bgr(out_path, img)
                rows.append((str(id_str), cam_key, int(decoded_idx), tick_ms, actual_time_ms, int(decoded_idx), out_path))

            # Small speed-up: if we passed all wanted indices, stop
            next_wanted_i += 1
            if next_wanted_i >= len(wanted_indices):
                break

        if decoded_idx > last_wanted or next_wanted_i >= len(wanted_indices):
            break

    container.close()

    # If video ended early, report missing indices
    if decoded_idx < last_wanted:
        missing = [i for i in wanted_indices if i > decoded_idx]
        if missing:
            print(f"[warn] {cam_key}: video ended at frame {decoded_idx}, missing {len(missing)} target frames (e.g. {missing[:10]})")

    return rows


# --------------------- Main ---------------------

def main() -> None:
    args = parse_args()

    root_dir: Path = args.root_dir
    video_dir = root_dir / args.video_subdir
    meta_dir = root_dir / args.meta_subdir
    frame_times_path = meta_dir / args.frame_times
    session_meta_path = meta_dir / args.session_meta

    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"--root-dir not found or not a directory: {root_dir}")
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(f"Video dir not found: {video_dir}")
    if not frame_times_path.exists():
        raise FileNotFoundError(f"frame_times.csv not found: {frame_times_path}")

    # Load session_meta.json if present (more exact slug -> video path mapping)
    meta = load_session_meta(session_meta_path)
    video_map = build_video_map_from_meta(root_dir, meta, video_dir)

    # Discover cameras from CSV (*_out_idx)
    cams_in_csv = set(discover_cam_keys_from_csv(frame_times_path, suffix="_out_idx"))
    if not cams_in_csv:
        raise RuntimeError(f"No '*_out_idx' camera columns found in: {frame_times_path}")

    # Cameras that actually have a video file
    cams_with_video = {c for c in cams_in_csv if c in video_map and video_map[c].exists()}

    if not cams_with_video:
        raise RuntimeError(
            "No cameras overlap between CSV '*_out_idx' columns and existing video files.\n"
            f"  CSV cams: {sorted(cams_in_csv)}\n"
            f"  Video map cams: {sorted(video_map.keys())}"
        )

    # Parse --cams
    cam_keys = parse_camera_selection(args.cams, cams_with_video)

    # Read targets from Recorder CSV
    _id_order, targets_by_cam = read_targets_from_recorder_csv(
        frame_times_path=frame_times_path,
        id_key=args.id_key,
        allowed_cam_keys=set(cam_keys),
    )

    # Output root
    args.out.mkdir(parents=True, exist_ok=True)

    # Extract per camera
    all_rows: List[Tuple[str, str, int, Optional[int], float, int, Path]] = []
    for cam in sorted(targets_by_cam.keys()):
        vp = video_map.get(cam)
        if vp is None or not vp.exists():
            print(f"[warn] missing video for cam '{cam}', skipping")
            continue

        rows = extract_for_one_camera_by_index(
            video_path=vp,
            cam_key=cam,
            targets=targets_by_cam[cam],
            out_root=args.out,
            ext=args.ext,
            name_with_time=args.name_with_time,
        )
        all_rows.extend(rows)

    # Write index CSV
    index_path = args.out / "frames_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "camera", "frame_index", "tick_video_ms", "actual_time_ms", "decoded_frame_index", "path"])
        for (id_str, cam, frame_idx, tick_ms, act_ms, dec_idx, path) in all_rows:
            w.writerow([
                id_str,
                cam,
                frame_idx,
                "" if tick_ms is None else int(tick_ms),
                "" if not np.isfinite(act_ms) else f"{act_ms:.3f}",
                dec_idx,
                str(path),
            ])

    ids_done = len(set(r[0] for r in all_rows))
    cams_done = len(set(r[1] for r in all_rows))
    print(f"[done] IDs processed: {ids_done} | cameras written: {cams_done}")
    print(f"[done] Output root: {args.out.resolve()}")
    print(f"[done] Index CSV:   {index_path.resolve()}")


if __name__ == "__main__":
    main()
