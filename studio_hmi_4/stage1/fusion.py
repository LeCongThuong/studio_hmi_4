"""Specialized hand-fusion helpers for stage-1 inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from studio_hmi_4.common.cv2_compat import require_cv2

if TYPE_CHECKING:  # pragma: no cover
    from .runner import Demo2Config


# MANO/OpenPose(21) -> MHR70 right-hand indices.
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


def extract_k70_keypoints(
    output_dict: Mapping[str, object],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if "pred_keypoints_2d" not in output_dict:
        return None, None
    arr = np.asarray(output_dict["pred_keypoints_2d"], dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (70, 2):
        return arr.copy(), arr
    if arr.ndim == 3 and arr.shape[-2:] == (70, 2):
        return arr[0].copy(), arr
    return None, arr


def write_back_k70_keypoints(
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


def build_specialized_hand_estimator(config: "Demo2Config") -> Optional[Any]:
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


def bbox_area(bbox: np.ndarray) -> float:
    if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
        return -1.0
    x0, y0, x1, y1 = bbox[:4]
    return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def extract_hand_candidates_from_wilor_outputs(
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
                "bbox_area": bbox_area(bbox),
            }
        )
    return candidates


def extract_hand_candidates_from_serialized(
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
                "bbox_area": bbox_area(bbox),
            }
        )
    return candidates


def find_precomputed_hand_file(
    hand_root: Path,
    rel_dir: str,
    image_name: str,
) -> Optional[Path]:
    base = hand_root / rel_dir
    for ext in (".npy", ".npz", ".json"):
        path = base / f"{image_name}{ext}"
        if path.is_file():
            return path
    return None


def load_precomputed_hand_payload(path: Path) -> Dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        return dict(obj) if isinstance(obj, Mapping) else {}

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
        archive = np.load(path, allow_pickle=True)
        if "detections" in archive:
            det_obj = archive["detections"]
            if isinstance(det_obj, np.ndarray) and det_obj.shape == () and hasattr(det_obj, "item"):
                obj = det_obj.item()
                if isinstance(obj, Mapping):
                    return dict(obj)
                if isinstance(obj, list):
                    return {"detections": obj}
            return {"detections": det_obj.tolist() if hasattr(det_obj, "tolist") else []}
        if "arr_0" in archive:
            arr0 = archive["arr_0"]
            if isinstance(arr0, np.ndarray) and arr0.shape == () and hasattr(arr0, "item"):
                obj = arr0.item()
                if isinstance(obj, Mapping):
                    return dict(obj)
        return {}

    return {}


def pick_best_hand_candidate(
    side_candidates: Sequence[Dict[str, object]],
    sam_wrist_xy: np.ndarray,
    wrist_max_dist_px: float,
) -> Optional[Dict[str, object]]:
    if len(side_candidates) == 0:
        return None

    wrist_ok = bool(np.isfinite(sam_wrist_xy).all())
    best = None
    best_score = float("inf")

    for candidate in side_candidates:
        cand_kpts = np.asarray(candidate["keypoints_2d"], dtype=np.float32)
        cand_wrist = cand_kpts[0]
        if not np.isfinite(cand_wrist).all():
            continue

        if wrist_ok:
            dist = float(np.linalg.norm(cand_wrist - sam_wrist_xy))
            if wrist_max_dist_px > 0.0 and dist > wrist_max_dist_px:
                continue
            score = dist
        else:
            score = float(-float(candidate.get("bbox_area", -1.0)))

        if score < best_score:
            best_score = score
            best = candidate

    return best


def apply_hand_candidates_to_output(
    output_dict: Dict[str, object],
    candidates: Sequence[Dict[str, object]],
    config: "Demo2Config",
    model_tag: str,
) -> Dict[str, object]:
    k70, original_arr = extract_k70_keypoints(output_dict)
    if k70 is None or original_arr is None:
        return output_dict

    if not isinstance(candidates, (list, tuple)) or len(candidates) == 0:
        output_dict["specialized_hand_used"] = 0
        output_dict["specialized_hand_num_points"] = 0
        output_dict["specialized_hand_status_right"] = "missing"
        output_dict["specialized_hand_status_left"] = "missing"
        output_dict["specialized_hand_model"] = str(model_tag)
        return output_dict

    right_candidates = [candidate for candidate in candidates if bool(candidate["is_right"])]
    left_candidates = [candidate for candidate in candidates if not bool(candidate["is_right"])]

    right_sel = pick_best_hand_candidate(
        side_candidates=right_candidates,
        sam_wrist_xy=np.asarray(k70[RIGHT_WRIST_MHR70_IDX], dtype=np.float32),
        wrist_max_dist_px=float(config.specialized_hand_wrist_max_dist_px),
    )
    left_sel = pick_best_hand_candidate(
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

    write_back_k70_keypoints(output_dict, original_arr=original_arr, k70=k70)
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


def apply_specialized_hand_fusion_live(
    output_dict: Dict[str, object],
    image_bgr: np.ndarray,
    hand_estimator: Any,
    config: "Demo2Config",
) -> Dict[str, object]:
    try:
        detect_rets = hand_estimator.predict(
            image_bgr,
            hand_conf=float(config.specialized_hand_detector_conf),
            rescale_factor=float(config.specialized_hand_rescale_factor),
        )
    except Exception:
        return output_dict

    candidates = extract_hand_candidates_from_wilor_outputs(
        detect_rets if isinstance(detect_rets, (list, tuple)) else []
    )
    return apply_hand_candidates_to_output(
        output_dict=output_dict,
        candidates=candidates,
        config=config,
        model_tag=str(config.specialized_hand_model),
    )


def apply_specialized_hand_fusion_precomputed(
    output_dict: Dict[str, object],
    hand_root: Path,
    rel_dir: str,
    image_name: str,
    config: "Demo2Config",
) -> Dict[str, object]:
    hand_file = find_precomputed_hand_file(
        hand_root=hand_root,
        rel_dir=rel_dir,
        image_name=image_name,
    )
    if hand_file is None:
        return apply_hand_candidates_to_output(
            output_dict=output_dict,
            candidates=[],
            config=config,
            model_tag=f"{config.specialized_hand_model}_precomputed",
        )
    payload = load_precomputed_hand_payload(hand_file)
    detections_obj = payload.get("detections", [])
    if not isinstance(detections_obj, (list, tuple)):
        detections_obj = []
    candidates = extract_hand_candidates_from_serialized(detections_obj)
    out = apply_hand_candidates_to_output(
        output_dict=output_dict,
        candidates=candidates,
        config=config,
        model_tag=f"{config.specialized_hand_model}_precomputed",
    )
    out["specialized_hand_input_file"] = str(hand_file)
    return out


def render_specialized_hand_debug_vis(
    image_bgr: np.ndarray,
    pre_k70: np.ndarray,
    post_k70: np.ndarray,
    source_mask: np.ndarray,
    output_dict: Mapping[str, object],
) -> np.ndarray:
    cv2 = require_cv2("stage-1 specialized-hand debug visualization")
    vis = image_bgr.copy()
    hand_indices = np.concatenate([MANO_TO_MHR70_RIGHT, MANO_TO_MHR70_LEFT]).astype(np.int64)
    hand_indices = np.unique(hand_indices)

    for idx in hand_indices.tolist():
        point = np.asarray(pre_k70[idx], dtype=np.float32).reshape(2)
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(vis, (x, y), 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    for idx in hand_indices.tolist():
        point = np.asarray(post_k70[idx], dtype=np.float32).reshape(2)
        if not np.isfinite(point).all():
            continue
        from_specialized = bool(idx < source_mask.size and int(source_mask[idx]) == 1)
        color = (0, 220, 0) if from_specialized else (0, 165, 255)
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
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
