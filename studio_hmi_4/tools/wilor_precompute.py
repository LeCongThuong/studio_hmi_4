#!/usr/bin/env python3
"""Run WiLoR-mini on image frames and export per-image hand detections.

Output format (per image):
  <output_root>/<rel_dir>/<image_stem>.npy (dict, allow_pickle=True)
  {
    "image_path": "...",
    "rel_dir": "...",
    "image_name": "...",
    "detections": [
      {
        "is_right": 0|1,
        "hand_bbox": [x0,y0,x1,y1],
        "pred_keypoints_2d": (21,2) float32
      },
      ...
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from studio_hmi_4.common import normalize_rel_dir as _normalize_rel_dir, relative_dir as _relative_dir

IMAGE_EXTENSIONS: Sequence[str] = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.gif",
    "*.bmp",
    "*.tiff",
    "*.webp",
)


def collect_images(image_root: str) -> List[str]:
    def _natural_key(p: str):
        rel = str(Path(p).resolve().relative_to(Path(image_root).resolve())).replace("\\", "/")
        parts = re.split(r"(\d+)", rel)
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
        for image in glob(str(Path(image_root) / "**" / ext), recursive=True)
    ]
    return sorted(images, key=_natural_key)


def _filter_images_by_rel_dirs(images: List[str], image_root: str, rel_dirs: Sequence[str]) -> List[str]:
    allowed = {_normalize_rel_dir(r) for r in rel_dirs}
    out: List[str] = []
    for image_path in images:
        rel_dir = _normalize_rel_dir(_relative_dir(image_path, image_root))
        if rel_dir in allowed:
            out.append(image_path)
    return out


def _serialize_detections(detect_rets: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for det in detect_rets:
        if not isinstance(det, dict):
            continue
        preds = det.get("wilor_preds", None)
        if not isinstance(preds, dict):
            continue
        k2d = np.asarray(preds.get("pred_keypoints_2d", None), dtype=np.float32)
        if k2d.ndim == 3 and k2d.shape[0] > 0:
            k2d = k2d[0]
        if k2d.shape != (21, 2):
            continue
        bbox = np.asarray(det.get("hand_bbox", []), dtype=np.float32).reshape(-1)
        bbox4 = bbox[:4] if bbox.size >= 4 else np.zeros((4,), dtype=np.float32)
        is_right = int(det.get("is_right", 1))
        out.append(
            {
                "is_right": int(is_right == 1),
                "hand_bbox": bbox4.astype(np.float32),
                "pred_keypoints_2d": k2d.astype(np.float32),
            }
        )
    return out


def _load_saved_payload_npy(path: Path) -> Dict[str, object]:
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception:
        return {}
    if isinstance(arr, np.ndarray) and arr.shape == () and hasattr(arr, "item"):
        obj = arr.item()
        if isinstance(obj, dict):
            return dict(obj)
    if isinstance(arr, dict):
        return dict(arr)
    return {}


def _render_debug_vis(
    image_bgr: np.ndarray,
    detections: Sequence[Dict[str, object]],
):
    import cv2

    vis = image_bgr.copy()
    for hand_idx, det in enumerate(detections):
        is_right = bool(int(det.get("is_right", 1)))
        color = (0, 180, 255) if is_right else (255, 120, 0)
        side = "R" if is_right else "L"

        bbox = np.asarray(det.get("hand_bbox", []), dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.isfinite(bbox[:4]).all():
            x0, y0, x1, y1 = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(
                vis,
                f"{side}{hand_idx}",
                (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

        k2d = np.asarray(det.get("pred_keypoints_2d", []), dtype=np.float32)
        if k2d.ndim == 3 and k2d.shape[0] > 0:
            k2d = k2d[0]
        if k2d.ndim == 2 and k2d.shape[1] == 2:
            for p in k2d:
                if not np.isfinite(p).all():
                    continue
                x, y = int(round(float(p[0]))), int(round(float(p[1])))
                cv2.circle(vis, (x, y), 2, color, -1, lineType=cv2.LINE_AA)

    if len(detections) == 0:
        cv2.putText(
            vis,
            "No hands detected",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return vis


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("WiLoR precompute export")
    ap.add_argument("--image_folder", required=True, type=str)
    ap.add_argument("--output_root", required=True, type=str)
    ap.add_argument("--include_rel_dirs", nargs="*", default=None)
    ap.add_argument("--cams", nargs="*", default=None, help="Optional camera stems filter, e.g. left front right.")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--hand_conf", type=float, default=0.3)
    ap.add_argument("--rescale_factor", type=float, default=2.5)
    ap.add_argument("--wilor_pretrained_dir", type=str, default="")
    ap.add_argument("--wilor_repo_id", type=str, default="warmshao/WiLoR-mini")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument(
        "--debug_vis",
        action="store_true",
        default=False,
        help="Save WiLoR debug visualization images with hand bbox/keypoints.",
    )
    ap.add_argument(
        "--debug_vis_root",
        type=str,
        default="",
        help="Optional root folder for debug images (default: <output_root>/debug_vis).",
    )
    ap.add_argument("--quiet", action="store_true", default=False)
    return ap.parse_args(argv)


def main(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_args()

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        def tqdm(it, **kwargs):  # type: ignore
            return it

    try:
        import cv2
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import WiLoR-mini. Run this script in the WiLoR environment."
        ) from exc

    image_root = Path(args.image_folder).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    debug_vis_root: Optional[Path] = None
    if bool(args.debug_vis):
        debug_vis_root = (
            Path(str(args.debug_vis_root).strip()).expanduser().resolve()
            if str(args.debug_vis_root).strip() != ""
            else (output_root / "debug_vis").resolve()
        )
        debug_vis_root.mkdir(parents=True, exist_ok=True)

    device_name = str(args.device).strip().lower()
    if device_name == "cuda" and (not torch.cuda.is_available()):
        device_name = "cpu"

    wilor_kwargs: Dict[str, object] = {
        "device": torch.device(device_name),
        "verbose": not bool(args.quiet),
    }
    if str(args.wilor_pretrained_dir).strip():
        wilor_kwargs["wilor_pretrained_dir"] = str(args.wilor_pretrained_dir).strip()
    if str(args.wilor_repo_id).strip():
        wilor_kwargs["WILOR_MINI_REPO_ID"] = str(args.wilor_repo_id).strip()

    pipeline = WiLorHandPose3dEstimationPipeline(**wilor_kwargs)

    images = collect_images(str(image_root))
    if args.include_rel_dirs:
        images = _filter_images_by_rel_dirs(images, str(image_root), args.include_rel_dirs)
    if args.cams:
        cams_set = {str(c).strip() for c in args.cams if str(c).strip() != ""}
        images = [p for p in images if Path(p).stem in cams_set]

    if len(images) == 0:
        print(f"[WARN] No images found in {image_root}")
        return 0

    num_done = 0
    num_skipped = 0
    num_errors = 0
    for image_path in tqdm(images, disable=bool(args.quiet)):
        rel_dir = _relative_dir(image_path, str(image_root))
        image_name = Path(image_path).stem
        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npy = out_dir / f"{image_name}.npy"
        debug_vis_path: Optional[Path] = None
        if debug_vis_root is not None:
            vis_dir = (debug_vis_root / rel_dir).resolve()
            vis_dir.mkdir(parents=True, exist_ok=True)
            debug_vis_path = vis_dir / f"{image_name}.jpg"

        if out_npy.exists() and (not bool(args.overwrite)):
            if debug_vis_path is not None and (not debug_vis_path.exists()):
                img = cv2.imread(image_path)
                if img is None:
                    num_errors += 1
                    num_skipped += 1
                    continue
                payload = _load_saved_payload_npy(out_npy)
                detections_obj = payload.get("detections", [])
                detections = detections_obj if isinstance(detections_obj, list) else []
                vis = _render_debug_vis(img, detections=detections)
                cv2.imwrite(str(debug_vis_path), vis)
            num_skipped += 1
            continue

        img = cv2.imread(image_path)
        if img is None:
            num_errors += 1
            continue

        try:
            detect_rets = pipeline.predict(
                img,
                hand_conf=float(args.hand_conf),
                rescale_factor=float(args.rescale_factor),
            )
        except Exception:
            detect_rets = []
            num_errors += 1

        detections = _serialize_detections(detect_rets if isinstance(detect_rets, (list, tuple)) else [])
        payload = {
            "image_path": str(Path(image_path).resolve()),
            "rel_dir": str(rel_dir),
            "image_name": str(image_name),
            "detections": detections,
        }
        np.save(out_npy, payload, allow_pickle=True)
        if debug_vis_path is not None:
            vis = _render_debug_vis(img, detections=detections)
            cv2.imwrite(str(debug_vis_path), vis)
        num_done += 1

    meta = {
        "image_folder": str(image_root),
        "output_root": str(output_root),
        "debug_vis": bool(args.debug_vis),
        "debug_vis_root": (None if debug_vis_root is None else str(debug_vis_root)),
        "num_images_total": len(images),
        "num_done": int(num_done),
        "num_skipped": int(num_skipped),
        "num_errors": int(num_errors),
        "device": str(device_name),
        "hand_conf": float(args.hand_conf),
        "rescale_factor": float(args.rescale_factor),
        "wilor_repo_id": str(args.wilor_repo_id),
        "wilor_pretrained_dir": str(args.wilor_pretrained_dir),
    }
    (output_root / "wilor_precompute_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(
        f"[WiLoR precompute] done={num_done} skipped={num_skipped} "
        f"errors={num_errors} total={len(images)} output={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
