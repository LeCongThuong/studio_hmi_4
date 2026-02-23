# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Stage 1: Run SAM-3D inference on image folders and persist per-frame artifacts.

Core idea:
1. Build a SAM-3D estimator (pose model + optional detector/segmentor/FOV models).
2. Run per-image inference to produce a prediction dict.
3. Save canonical `.npy` outputs used by downstream triangulation/optimization stages.
4. Optionally save render/mesh debug outputs and extracted MHR parameter sidecars.

This module is intentionally usable in two ways:
- CLI script for quick debugging.
- Importable API (`Demo2Config`, `run_demo`) for orchestration in a larger pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from tools.utils import save_mesh_results
from tqdm import tqdm

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

# MANO/OpenPose(21) -> MHR70 right-hand indices.
# Source mapping rationale is documented in hand_mapping.md.
MANO_TO_MHR70_RIGHT = np.array(
    [
        41, 24, 23, 22, 21,
        28, 27, 26, 25,
        32, 31, 30, 29,
        36, 35, 34, 33,
        40, 39, 38, 37,
    ],
    dtype=np.int64,
)

# MANO/OpenPose(21) -> MHR70 left-hand indices.
MANO_TO_MHR70_LEFT = np.array(
    [
        62, 45, 44, 43, 42,
        49, 48, 47, 46,
        53, 52, 51, 50,
        57, 56, 55, 54,
        61, 60, 59, 58,
    ],
    dtype=np.int64,
)

RIGHT_WRIST_MHR70_IDX = 41
LEFT_WRIST_MHR70_IDX = 62


@dataclass
class Demo2Config:
    """Configuration for stage-1 SAM-3D inference."""

    image_folder: str
    checkpoint_path: str
    output_folder: str = ""
    detector_name: str = "vitdet"
    segmentor_name: str = "sam2"
    fov_name: str = "moge2"
    detector_path: str = ""
    segmentor_path: str = ""
    fov_path: str = ""
    mhr_path: str = ""
    bbox_thresh: float = 0.8
    use_mask: bool = False
    debug: bool = False
    save_mhr_params: bool = False
    include_rel_dirs: Optional[List[str]] = None
    person_select_strategy: str = "largest_bbox"
    person_index: int = 0
    enable_specialized_hand_fusion: bool = False
    specialized_hand_source: str = "precomputed"
    specialized_hand_model: str = "none"
    specialized_hand_input_root: str = ""
    specialized_hand_device: str = "cuda"
    specialized_hand_detector_conf: float = 0.3
    specialized_hand_rescale_factor: float = 2.5
    specialized_hand_wrist_max_dist_px: float = 140.0
    replace_wrist_with_specialized: bool = False
    specialized_hand_debug_vis: bool = False
    specialized_hand_debug_dirname: str = "specialized_hand_debug"
    specialized_hand_verbose: bool = False
    wilor_pretrained_dir: str = ""
    wilor_repo_id: str = "warmshao/WiLoR-mini"


@dataclass
class FrameResult:
    """Per-image output summary returned by `run_demo`."""

    image_path: Path
    rel_dir: str
    npy_path: Optional[Path]
    mhr_params_path: Optional[Path]
    has_prediction: bool


@dataclass
class Demo2RunResult:
    """Aggregated stage-1 output locations and frame-level status."""

    output_root: Path
    npy_root: Path
    render_root: Path
    mesh_root: Path
    mhr_params_root: Optional[Path]
    frames: List[FrameResult]


def resolve_output_folder(image_folder: str, output_folder: str) -> Path:
    """Resolve output directory, defaulting to `./output/<image_folder_name>`."""

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


def resolve_model_paths(config: Demo2Config) -> tuple[str, str, str, str]:
    mhr_path = config.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = config.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = config.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = config.fov_path or os.environ.get("SAM3D_FOV_PATH", "")
    return mhr_path, detector_path, segmentor_path, fov_path


def build_estimator(config: Demo2Config) -> SAM3DBodyEstimator:
    """Construct the SAM-3D estimator and optional auxiliary models."""

    mhr_path, detector_path, segmentor_path, fov_path = resolve_model_paths(config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = load_sam_3d_body(
        config.checkpoint_path,
        device=device,
        mhr_path=mhr_path,
    )

    human_detector = None
    human_segmentor = None
    fov_estimator = None

    if config.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=config.detector_name,
            device=device,
            path=detector_path,
        )

    should_build_segmentor = (
        config.segmentor_name != "sam2"
        or (config.segmentor_name == "sam2" and len(segmentor_path) > 0)
    )
    if should_build_segmentor:
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=config.segmentor_name,
            device=device,
            path=segmentor_path,
        )

    if config.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=config.fov_name,
            device=device,
            path=fov_path,
        )

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


def collect_images(image_root: str) -> List[str]:
    def _natural_key(p: str):
        rel = os.path.relpath(p, image_root)
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
        candidates = [str(p) for p in ply_files if p is not None]
    else:
        candidates = [str(ply_files)]

    return [p for p in candidates if os.path.isfile(p) and p.lower().endswith(".ply")]


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

    candidates = [o for o in outputs if isinstance(o, Mapping)]
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
    np.save(out_path, dict(data), allow_pickle=True)
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


def _relative_dir(image_path: str, image_root: str) -> str:
    rel_dir = os.path.relpath(os.path.dirname(image_path), image_root)
    if rel_dir == ".":
        return ""
    return rel_dir


def _normalize_rel_dir(rel_dir: str) -> str:
    norm = rel_dir.replace("\\", "/").strip("/")
    return norm


def _filter_images_by_rel_dirs(images: List[str], image_root: str, rel_dirs: Sequence[str]) -> List[str]:
    allowed = {_normalize_rel_dir(r) for r in rel_dirs}
    filtered: List[str] = []
    for image_path in images:
        rel_dir = _normalize_rel_dir(_relative_dir(image_path, image_root))
        if rel_dir in allowed:
            filtered.append(image_path)
    return filtered


def _extract_k70_keypoints(
    output_dict: Mapping[str, object],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return `(k70, original_array)` from `pred_keypoints_2d` if shape is compatible."""

    if "pred_keypoints_2d" not in output_dict:
        return None, None
    arr = np.asarray(output_dict["pred_keypoints_2d"], dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (70, 2):
        return arr.copy(), arr
    if arr.ndim == 3 and arr.shape[-2:] == (70, 2):
        return arr[0].copy(), arr
    return None, arr


def _write_back_k70_keypoints(
    output_dict: Dict[str, object],
    original_arr: np.ndarray,
    k70: np.ndarray,
) -> None:
    k70 = np.asarray(k70, dtype=np.float32)
    if original_arr.ndim == 2:
        output_dict["pred_keypoints_2d"] = k70
    elif original_arr.ndim == 3:
        out = np.asarray(original_arr, dtype=np.float32).copy()
        out[0] = k70
        output_dict["pred_keypoints_2d"] = out


def _build_specialized_hand_estimator(config: Demo2Config) -> Optional[Any]:
    if not bool(config.enable_specialized_hand_fusion):
        return None

    source = str(config.specialized_hand_source).strip().lower()
    if source in {"precomputed", "file", "files"}:
        return None

    model_name = str(config.specialized_hand_model).strip().lower()
    if model_name in {"", "none", "off", "disabled"}:
        return None

    if model_name != "wilor":
        raise ValueError(
            f"Unsupported specialized hand model '{config.specialized_hand_model}'. "
            "Supported values: none, wilor."
        )

    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
    except Exception as exc:
        raise ImportError(
            "Failed to import WiLoR-mini in live mode. "
            "Either install WiLoR-mini or use --specialized_hand_source precomputed."
        ) from exc

    requested_device = str(config.specialized_hand_device).strip().lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"

    wilor_kwargs: Dict[str, Any] = {
        "device": torch.device(requested_device),
        "verbose": bool(config.specialized_hand_verbose),
    }
    if str(config.wilor_pretrained_dir).strip():
        wilor_kwargs["wilor_pretrained_dir"] = str(config.wilor_pretrained_dir).strip()
    if str(config.wilor_repo_id).strip():
        wilor_kwargs["WILOR_MINI_REPO_ID"] = str(config.wilor_repo_id).strip()

    return WiLorHandPose3dEstimationPipeline(**wilor_kwargs)


def _bbox_area(bbox: np.ndarray) -> float:
    if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
        return -1.0
    x0, y0, x1, y1 = bbox[:4]
    return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def _extract_hand_candidates_from_wilor_outputs(
    detect_rets: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for det in detect_rets:
        preds = det.get("wilor_preds", None)
        if not isinstance(preds, Mapping):
            continue
        k2d = np.asarray(preds.get("pred_keypoints_2d", None), dtype=np.float32)
        if k2d.ndim == 3 and k2d.shape[0] > 0:
            k2d = k2d[0]
        if k2d.shape != (21, 2):
            continue
        bbox = np.asarray(det.get("hand_bbox", []), dtype=np.float32).reshape(-1)
        is_right = int(det.get("is_right", 1))
        candidates.append(
            {
                "is_right": bool(is_right == 1),
                "keypoints_2d": k2d.astype(np.float32),
                "bbox": bbox.astype(np.float32),
                "bbox_area": _bbox_area(bbox),
            }
        )
    return candidates


def _extract_hand_candidates_from_serialized(
    detections: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for det in detections:
        if not isinstance(det, Mapping):
            continue
        k2d_obj = det.get("pred_keypoints_2d", det.get("keypoints_2d", None))
        if k2d_obj is None:
            continue
        k2d = np.asarray(k2d_obj, dtype=np.float32)
        if k2d.ndim == 3 and k2d.shape[0] > 0:
            k2d = k2d[0]
        if k2d.shape != (21, 2):
            continue
        bbox_obj = det.get("hand_bbox", det.get("bbox", []))
        bbox = np.asarray(bbox_obj, dtype=np.float32).reshape(-1)
        is_right = int(det.get("is_right", 1))
        candidates.append(
            {
                "is_right": bool(is_right == 1),
                "keypoints_2d": k2d.astype(np.float32),
                "bbox": bbox.astype(np.float32),
                "bbox_area": _bbox_area(bbox),
            }
        )
    return candidates


def _find_precomputed_hand_file(
    hand_root: Path,
    rel_dir: str,
    image_name: str,
) -> Optional[Path]:
    base = hand_root / rel_dir
    for ext in (".npy", ".npz", ".json"):
        p = base / f"{image_name}{ext}"
        if p.is_file():
            return p
    return None


def _load_precomputed_hand_payload(path: Path) -> Dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, Mapping):
            return dict(obj)
        return {}

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.shape == () and hasattr(arr, "item"):
            obj = arr.item()
            if isinstance(obj, Mapping):
                return dict(obj)
        if isinstance(arr, Mapping):
            return dict(arr)
        return {}

    if suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        if "detections" in z:
            det_obj = z["detections"]
            if isinstance(det_obj, np.ndarray) and det_obj.shape == () and hasattr(det_obj, "item"):
                obj = det_obj.item()
                if isinstance(obj, Mapping):
                    return dict(obj)
                if isinstance(obj, list):
                    return {"detections": obj}
            return {"detections": det_obj.tolist() if hasattr(det_obj, "tolist") else []}
        if "arr_0" in z:
            arr0 = z["arr_0"]
            if isinstance(arr0, np.ndarray) and arr0.shape == () and hasattr(arr0, "item"):
                obj = arr0.item()
                if isinstance(obj, Mapping):
                    return dict(obj)
        return {}

    return {}


def _pick_best_hand_candidate(
    side_candidates: Sequence[Dict[str, object]],
    sam_wrist_xy: np.ndarray,
    wrist_max_dist_px: float,
) -> Optional[Dict[str, object]]:
    if len(side_candidates) == 0:
        return None

    wrist_ok = bool(np.isfinite(sam_wrist_xy).all())
    best = None
    best_score = float("inf")

    for cand in side_candidates:
        cand_kpts = np.asarray(cand["keypoints_2d"], dtype=np.float32)
        cand_wrist = cand_kpts[0]
        if not np.isfinite(cand_wrist).all():
            continue

        if wrist_ok:
            dist = float(np.linalg.norm(cand_wrist - sam_wrist_xy))
            if wrist_max_dist_px > 0.0 and dist > wrist_max_dist_px:
                continue
            score = dist
        else:
            # If SAM wrist is not finite, prefer larger bbox as a weak prior.
            score = float(-float(cand.get("bbox_area", -1.0)))

        if score < best_score:
            best_score = score
            best = cand

    return best


def _apply_hand_candidates_to_output(
    output_dict: Dict[str, object],
    candidates: Sequence[Dict[str, object]],
    config: Demo2Config,
    model_tag: str,
) -> Dict[str, object]:
    k70, original_arr = _extract_k70_keypoints(output_dict)
    if k70 is None or original_arr is None:
        return output_dict

    if not isinstance(candidates, (list, tuple)) or len(candidates) == 0:
        output_dict["specialized_hand_used"] = 0
        output_dict["specialized_hand_num_points"] = 0
        output_dict["specialized_hand_status_right"] = "missing"
        output_dict["specialized_hand_status_left"] = "missing"
        output_dict["specialized_hand_model"] = str(model_tag)
        return output_dict

    right_candidates = [c for c in candidates if bool(c["is_right"])]
    left_candidates = [c for c in candidates if not bool(c["is_right"])]

    right_sel = _pick_best_hand_candidate(
        side_candidates=right_candidates,
        sam_wrist_xy=np.asarray(k70[RIGHT_WRIST_MHR70_IDX], dtype=np.float32),
        wrist_max_dist_px=float(config.specialized_hand_wrist_max_dist_px),
    )
    left_sel = _pick_best_hand_candidate(
        side_candidates=left_candidates,
        sam_wrist_xy=np.asarray(k70[LEFT_WRIST_MHR70_IDX], dtype=np.float32),
        wrist_max_dist_px=float(config.specialized_hand_wrist_max_dist_px),
    )

    source_mask = np.zeros((70,), dtype=np.int8)
    replace_from = 0 if bool(config.replace_wrist_with_specialized) else 1
    right_replaced = 0
    left_replaced = 0

    if right_sel is not None:
        k2d = np.asarray(right_sel["keypoints_2d"], dtype=np.float32)
        for i in range(replace_from, 21):
            mhr_idx = int(MANO_TO_MHR70_RIGHT[i])
            if np.isfinite(k2d[i]).all():
                k70[mhr_idx] = k2d[i]
                source_mask[mhr_idx] = 1
                right_replaced += 1

    if left_sel is not None:
        k2d = np.asarray(left_sel["keypoints_2d"], dtype=np.float32)
        for i in range(replace_from, 21):
            mhr_idx = int(MANO_TO_MHR70_LEFT[i])
            if np.isfinite(k2d[i]).all():
                k70[mhr_idx] = k2d[i]
                source_mask[mhr_idx] = 1
                left_replaced += 1

    _write_back_k70_keypoints(output_dict, original_arr=original_arr, k70=k70)
    output_dict["pred_keypoints_2d_source"] = source_mask.astype(np.int8)
    output_dict["specialized_hand_model"] = str(model_tag)
    output_dict["specialized_hand_used"] = int((right_replaced + left_replaced) > 0)
    output_dict["specialized_hand_num_points"] = int(right_replaced + left_replaced)
    output_dict["specialized_hand_replace_wrist"] = int(bool(config.replace_wrist_with_specialized))
    output_dict["specialized_hand_status_right"] = (
        "specialized" if right_replaced > 0 else "sam_fallback"
    )
    output_dict["specialized_hand_status_left"] = (
        "specialized" if left_replaced > 0 else "sam_fallback"
    )
    return output_dict


def _apply_specialized_hand_fusion_live(
    output_dict: Dict[str, object],
    image_bgr: np.ndarray,
    hand_estimator: Any,
    config: Demo2Config,
) -> Dict[str, object]:
    try:
        detect_rets = hand_estimator.predict(
            image_bgr,
            hand_conf=float(config.specialized_hand_detector_conf),
            rescale_factor=float(config.specialized_hand_rescale_factor),
        )
    except Exception:
        return output_dict
    candidates = _extract_hand_candidates_from_wilor_outputs(
        detect_rets if isinstance(detect_rets, (list, tuple)) else []
    )
    return _apply_hand_candidates_to_output(
        output_dict=output_dict,
        candidates=candidates,
        config=config,
        model_tag=str(config.specialized_hand_model),
    )


def _apply_specialized_hand_fusion_precomputed(
    output_dict: Dict[str, object],
    hand_root: Path,
    rel_dir: str,
    image_name: str,
    config: Demo2Config,
) -> Dict[str, object]:
    hand_file = _find_precomputed_hand_file(
        hand_root=hand_root,
        rel_dir=rel_dir,
        image_name=image_name,
    )
    if hand_file is None:
        return _apply_hand_candidates_to_output(
            output_dict=output_dict,
            candidates=[],
            config=config,
            model_tag=f"{config.specialized_hand_model}_precomputed",
        )
    payload = _load_precomputed_hand_payload(hand_file)
    detections_obj = payload.get("detections", [])
    if not isinstance(detections_obj, (list, tuple)):
        detections_obj = []
    candidates = _extract_hand_candidates_from_serialized(detections_obj)
    out = _apply_hand_candidates_to_output(
        output_dict=output_dict,
        candidates=candidates,
        config=config,
        model_tag=f"{config.specialized_hand_model}_precomputed",
    )
    out["specialized_hand_input_file"] = str(hand_file)
    return out


def _render_specialized_hand_debug_vis(
    image_bgr: np.ndarray,
    pre_k70: np.ndarray,
    post_k70: np.ndarray,
    source_mask: np.ndarray,
    output_dict: Mapping[str, object],
) -> np.ndarray:
    vis = image_bgr.copy()
    hand_indices = np.concatenate([MANO_TO_MHR70_RIGHT, MANO_TO_MHR70_LEFT]).astype(np.int64)
    hand_indices = np.unique(hand_indices)

    # Background reference (SAM before fusion): white hollow points.
    for idx in hand_indices.tolist():
        p = np.asarray(pre_k70[idx], dtype=np.float32).reshape(2)
        if not np.isfinite(p).all():
            continue
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        cv2.circle(vis, (x, y), 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Final points after fusion: green = specialized, orange = SAM fallback.
    for idx in hand_indices.tolist():
        p = np.asarray(post_k70[idx], dtype=np.float32).reshape(2)
        if not np.isfinite(p).all():
            continue
        from_specialized = bool(idx < source_mask.size and int(source_mask[idx]) == 1)
        color = (0, 220, 0) if from_specialized else (0, 165, 255)
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        cv2.circle(vis, (x, y), 2, color, -1, lineType=cv2.LINE_AA)

    right_status = str(output_dict.get("specialized_hand_status_right", ""))
    left_status = str(output_dict.get("specialized_hand_status_left", ""))
    replaced = int(np.sum(source_mask[hand_indices] == 1)) if source_mask.size > 0 else 0

    cv2.putText(
        vis,
        f"R:{right_status}  L:{left_status}  replaced:{replaced}",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "white: SAM-before  green: specialized  orange: SAM-fallback",
        (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return vis


def run_demo(
    config: Demo2Config,
    estimator: Optional[SAM3DBodyEstimator] = None,
    show_progress: bool = True,
) -> Demo2RunResult:
    """Execute stage-1 inference for all images under `config.image_folder`.

    Returns a `Demo2RunResult` containing output roots and per-frame save status.
    """

    output_root = resolve_output_folder(config.image_folder, config.output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    render_root, npy_root, mesh_root, mhr_root = ensure_output_dirs(
        output_root=output_root,
        debug=config.debug,
        save_mhr_params=config.save_mhr_params,
    )
    if estimator is None:
        estimator = build_estimator(config)
    hand_estimator = _build_specialized_hand_estimator(config)
    hand_source = str(config.specialized_hand_source).strip().lower()
    precomputed_hand_root: Optional[Path] = None
    hand_debug_root: Optional[Path] = None
    if bool(config.enable_specialized_hand_fusion) and hand_source in {"precomputed", "file", "files"}:
        hand_input = str(config.specialized_hand_input_root).strip()
        if hand_input == "":
            raise ValueError(
                "specialized_hand_source is precomputed but --specialized_hand_input_root is empty."
            )
        precomputed_hand_root = Path(hand_input).expanduser().resolve()
        if not precomputed_hand_root.is_dir():
            raise FileNotFoundError(f"Precomputed hand input root not found: {precomputed_hand_root}")
    if bool(config.enable_specialized_hand_fusion) and bool(config.specialized_hand_debug_vis):
        debug_dirname = str(config.specialized_hand_debug_dirname).strip() or "specialized_hand_debug"
        hand_debug_root = (output_root / debug_dirname).resolve()
        hand_debug_root.mkdir(parents=True, exist_ok=True)

    images_list = collect_images(config.image_folder)
    if config.include_rel_dirs:
        images_list = _filter_images_by_rel_dirs(
            images=images_list,
            image_root=config.image_folder,
            rel_dirs=config.include_rel_dirs,
        )
    frames: List[FrameResult] = []

    if len(images_list) == 0:
        print(f"[WARN] No images found in {config.image_folder}")

    for image_path in tqdm(images_list, disable=not show_progress):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=config.bbox_thresh,
            use_mask=config.use_mask,
        )
        output_dict = extract_primary_output(
            outputs,
            strategy=config.person_select_strategy,
            person_index=config.person_index,
        )

        rel_dir = _relative_dir(image_path, config.image_folder)
        image_name = Path(image_path).stem

        pre_fusion_k70: Optional[np.ndarray] = None
        if output_dict is not None and bool(config.enable_specialized_hand_fusion):
            pre_fusion_k70, _ = _extract_k70_keypoints(output_dict)

        need_live_hand_image = bool(
            output_dict is not None
            and bool(config.enable_specialized_hand_fusion)
            and hand_source in {"live", "runtime"}
            and hand_estimator is not None
        )
        need_hand_debug_image = bool(
            output_dict is not None
            and bool(config.enable_specialized_hand_fusion)
            and bool(config.specialized_hand_debug_vis)
        )
        img_cv2 = None
        if output_dict is not None and (config.debug or need_live_hand_image or need_hand_debug_image):
            img_cv2 = cv2.imread(image_path)

        if output_dict is not None and bool(config.enable_specialized_hand_fusion):
            if hand_source in {"precomputed", "file", "files"}:
                assert precomputed_hand_root is not None
                output_dict = _apply_specialized_hand_fusion_precomputed(
                    output_dict=output_dict,
                    hand_root=precomputed_hand_root,
                    rel_dir=rel_dir,
                    image_name=image_name,
                    config=config,
                )
            elif hand_source in {"live", "runtime"} and hand_estimator is not None and img_cv2 is not None:
                output_dict = _apply_specialized_hand_fusion_live(
                    output_dict=output_dict,
                    image_bgr=img_cv2,
                    hand_estimator=hand_estimator,
                    config=config,
                )

        if (
            hand_debug_root is not None
            and output_dict is not None
            and img_cv2 is not None
            and pre_fusion_k70 is not None
        ):
            post_fusion_k70, _ = _extract_k70_keypoints(output_dict)
            if post_fusion_k70 is not None:
                source_mask = np.asarray(
                    output_dict.get("pred_keypoints_2d_source", np.zeros((70,), dtype=np.int8)),
                    dtype=np.int8,
                ).reshape(-1)
                vis = _render_specialized_hand_debug_vis(
                    image_bgr=img_cv2,
                    pre_k70=pre_fusion_k70,
                    post_k70=post_fusion_k70,
                    source_mask=source_mask,
                    output_dict=output_dict,
                )
                vis_out_dir = hand_debug_root / rel_dir
                vis_out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_out_dir / f"{image_name}.jpg"), vis)

        npy_out_dir = npy_root / rel_dir
        npy_out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = None
        if output_dict is not None:
            npy_path = save_dict_npy(output_dict, npy_out_dir, image_name)

        mhr_params_path = None
        if mhr_root is not None and output_dict is not None:
            mhr_out_dir = mhr_root / rel_dir
            mhr_out_dir.mkdir(parents=True, exist_ok=True)
            mhr_params = extract_mhr_params(output_dict, strict=False)
            if mhr_params:
                mhr_params_path = save_dict_npy(mhr_params, mhr_out_dir, image_name)

        if config.debug and output_dict is not None:
            render_out_dir = render_root / rel_dir
            mesh_out_dir = mesh_root / rel_dir
            render_out_dir.mkdir(parents=True, exist_ok=True)
            mesh_out_dir.mkdir(parents=True, exist_ok=True)

            if img_cv2 is not None:
                debug_outputs = (
                    outputs
                    if isinstance(outputs, (list, tuple))
                    else [output_dict]
                )
                ply_files = save_mesh_results(
                    img_cv2,
                    debug_outputs,
                    estimator.faces,
                    str(render_out_dir),
                    image_name,
                )
                move_mesh_files(ply_files, mesh_out_dir, image_name)

        frames.append(
            FrameResult(
                image_path=Path(image_path),
                rel_dir=rel_dir,
                npy_path=npy_path,
                mhr_params_path=mhr_params_path,
                has_prediction=output_dict is not None,
            )
        )

    meta = {
        "image_folder": str(Path(config.image_folder).expanduser().resolve()),
        "include_rel_dirs": list(config.include_rel_dirs) if config.include_rel_dirs else None,
        "checkpoint_path": str(config.checkpoint_path),
        "detector_name": str(config.detector_name),
        "segmentor_name": str(config.segmentor_name),
        "fov_name": str(config.fov_name),
        "person_select_strategy": str(config.person_select_strategy),
        "person_index": int(config.person_index),
        "enable_specialized_hand_fusion": bool(config.enable_specialized_hand_fusion),
        "specialized_hand_source": str(config.specialized_hand_source),
        "specialized_hand_model": str(config.specialized_hand_model),
        "specialized_hand_input_root": (
            None
            if precomputed_hand_root is None
            else str(precomputed_hand_root)
        ),
        "specialized_hand_detector_conf": float(config.specialized_hand_detector_conf),
        "specialized_hand_rescale_factor": float(config.specialized_hand_rescale_factor),
        "specialized_hand_wrist_max_dist_px": float(config.specialized_hand_wrist_max_dist_px),
        "replace_wrist_with_specialized": bool(config.replace_wrist_with_specialized),
        "specialized_hand_debug_vis": bool(config.specialized_hand_debug_vis),
        "specialized_hand_debug_dirname": str(config.specialized_hand_debug_dirname),
        "wilor_repo_id": str(config.wilor_repo_id),
    }
    (output_root / "stage1_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return Demo2RunResult(
        output_root=output_root,
        npy_root=npy_root,
        render_root=render_root,
        mesh_root=mesh_root,
        mhr_params_root=mhr_root,
        frames=frames,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for standalone stage-1 execution."""

    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_MHR_PATH)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="If set, save render + mesh into <output_folder>/{render,mesh}/. If not set, only .npy is saved.",
    )
    parser.add_argument(
        "--save_mhr_params",
        action="store_true",
        default=False,
        help="If set, save extracted MHR params to <output_folder>/mhr_params/<rel_path>/<image>.npy.",
    )
    parser.add_argument(
        "--include_rel_dirs",
        nargs="*",
        default=None,
        help="Optional relative subdirs under image_folder to process (e.g., 100 101).",
    )
    parser.add_argument(
        "--person_select_strategy",
        type=str,
        default="largest_bbox",
        choices=["first", "largest_bbox", "person_index"],
        help="How to pick a person when detector returns multiple outputs.",
    )
    parser.add_argument(
        "--person_index",
        type=int,
        default=0,
        help="Person index to use when --person_select_strategy=person_index.",
    )
    parser.add_argument(
        "--enable_specialized_hand_fusion",
        action="store_true",
        default=False,
        help="Enable specialized hand-model fusion into pred_keypoints_2d.",
    )
    parser.add_argument(
        "--specialized_hand_source",
        type=str,
        default="precomputed",
        choices=["precomputed", "live"],
        help="Source of specialized hand keypoints: precomputed files or live model inference.",
    )
    parser.add_argument(
        "--specialized_hand_model",
        type=str,
        default="none",
        choices=["none", "wilor"],
        help="Specialized hand model used for fusion.",
    )
    parser.add_argument(
        "--specialized_hand_input_root",
        type=str,
        default="",
        help=(
            "Root directory of precomputed hand detections. "
            "Expected files: <root>/<rel_dir>/<image_name>.npy|.npz|.json"
        ),
    )
    parser.add_argument(
        "--specialized_hand_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for specialized hand model inference.",
    )
    parser.add_argument(
        "--specialized_hand_detector_conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold for specialized hand detector.",
    )
    parser.add_argument(
        "--specialized_hand_rescale_factor",
        type=float,
        default=2.5,
        help="Hand crop rescale factor for specialized hand model.",
    )
    parser.add_argument(
        "--specialized_hand_wrist_max_dist_px",
        type=float,
        default=140.0,
        help="Reject specialized hand candidate if wrist is too far from SAM wrist (pixels).",
    )
    parser.add_argument(
        "--replace_wrist_with_specialized",
        action="store_true",
        default=False,
        help="Also replace wrist keypoint with specialized model (default keeps SAM wrist).",
    )
    parser.add_argument(
        "--specialized_hand_debug_vis",
        action="store_true",
        default=False,
        help="Save debug overlays for SAM-vs-specialized hand fusion.",
    )
    parser.add_argument(
        "--specialized_hand_debug_dirname",
        type=str,
        default="specialized_hand_debug",
        help="Subfolder under output root used for specialized hand fusion debug images.",
    )
    parser.add_argument(
        "--specialized_hand_verbose",
        action="store_true",
        default=False,
        help="Enable verbose logs from specialized hand model.",
    )
    parser.add_argument(
        "--wilor_pretrained_dir",
        type=str,
        default="",
        help="Optional local directory for WiLoR-mini pretrained assets.",
    )
    parser.add_argument(
        "--wilor_repo_id",
        type=str,
        default="warmshao/WiLoR-mini",
        help="Hugging Face repo id used by WiLoR-mini for pretrained assets.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> Demo2Config:
    """Convert parsed CLI args to `Demo2Config`."""

    return Demo2Config(
        image_folder=args.image_folder,
        checkpoint_path=args.checkpoint_path,
        output_folder=args.output_folder,
        detector_name=args.detector_name,
        segmentor_name=args.segmentor_name,
        fov_name=args.fov_name,
        detector_path=args.detector_path,
        segmentor_path=args.segmentor_path,
        fov_path=args.fov_path,
        mhr_path=args.mhr_path,
        bbox_thresh=args.bbox_thresh,
        use_mask=args.use_mask,
        debug=args.debug,
        save_mhr_params=args.save_mhr_params,
        include_rel_dirs=args.include_rel_dirs,
        person_select_strategy=args.person_select_strategy,
        person_index=args.person_index,
        enable_specialized_hand_fusion=args.enable_specialized_hand_fusion,
        specialized_hand_source=args.specialized_hand_source,
        specialized_hand_model=args.specialized_hand_model,
        specialized_hand_input_root=args.specialized_hand_input_root,
        specialized_hand_device=args.specialized_hand_device,
        specialized_hand_detector_conf=args.specialized_hand_detector_conf,
        specialized_hand_rescale_factor=args.specialized_hand_rescale_factor,
        specialized_hand_wrist_max_dist_px=args.specialized_hand_wrist_max_dist_px,
        replace_wrist_with_specialized=args.replace_wrist_with_specialized,
        specialized_hand_debug_vis=args.specialized_hand_debug_vis,
        specialized_hand_debug_dirname=args.specialized_hand_debug_dirname,
        specialized_hand_verbose=args.specialized_hand_verbose,
        wilor_pretrained_dir=args.wilor_pretrained_dir,
        wilor_repo_id=args.wilor_repo_id,
    )


def main(args: Optional[argparse.Namespace] = None) -> Demo2RunResult:
    """CLI/programmatic entrypoint for stage-1 inference."""

    if args is None:
        args = parse_args()
    config = namespace_to_config(args)
    return run_demo(config)


if __name__ == "__main__":
    main()
