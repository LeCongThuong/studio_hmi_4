"""Recovery, temporal-state, and metric-loading helpers for sequence runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from studio_hmi_4.stage2.runner import NP_EXTS, find_existing_with_exts

from .temporal import copy_frame_dict, interpolate_frame_dict, load_npy_dict, save_npy_dict
from .types import FramePipelineResult


def load_stage1_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.is_file():
        return None
    try:
        import json

        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def stage1_meta_matches(
    existing_meta: Optional[Dict[str, Any]],
    expected_meta: Dict[str, Any],
) -> bool:
    if existing_meta is None:
        return False
    keys = [
        "image_folder",
        "include_rel_dirs",
        "checkpoint_path",
        "detector_name",
        "segmentor_name",
        "fov_name",
        "person_select_strategy",
        "person_index",
        "enable_specialized_hand_fusion",
        "specialized_hand_source",
        "specialized_hand_model",
        "specialized_hand_input_root",
        "specialized_hand_detector_conf",
        "specialized_hand_rescale_factor",
        "specialized_hand_wrist_max_dist_px",
        "replace_wrist_with_specialized",
        "specialized_hand_debug_vis",
        "specialized_hand_debug_dirname",
        "wilor_repo_id",
    ]
    return all(existing_meta.get(key) == expected_meta.get(key) for key in keys)


def update_result_from_opt(dst: FramePipelineResult, opt_res: Any) -> None:
    dst.best_loss = float(opt_res.best_loss)
    dst.final_loss = float(opt_res.final_loss)
    dst.best_data_loss = float(opt_res.best_data_loss)
    dst.final_data_loss = float(opt_res.final_data_loss)
    dst.best_iter = int(opt_res.best_iter)
    dst.is_bad_loss = bool(opt_res.is_bad_loss)


def load_pose_if_good(opt_npy: Path) -> Optional[np.ndarray]:
    try:
        d = load_npy_dict(opt_npy)
    except Exception:
        return None
    is_bad = bool(int(np.asarray(d.get("opt_is_bad_loss", 0)).reshape(())))
    if is_bad or "body_pose_params" not in d:
        return None
    return np.asarray(d["body_pose_params"], dtype=np.float32).reshape(-1)


def load_similarity_if_good(opt_npy: Path) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
    try:
        d = load_npy_dict(opt_npy)
    except Exception:
        return None
    is_bad = bool(int(np.asarray(d.get("opt_is_bad_loss", 0)).reshape(())))
    if is_bad:
        return None
    if "opt_sim_scale" not in d or "opt_sim_R" not in d or "opt_sim_t" not in d:
        return None
    sim_scale = float(np.asarray(d["opt_sim_scale"]).reshape(()))
    sim_R = np.asarray(d["opt_sim_R"], dtype=np.float32).reshape(3, 3)
    sim_t = np.asarray(d["opt_sim_t"], dtype=np.float32).reshape(3)
    if not np.isfinite(sim_scale):
        return None
    if not np.isfinite(sim_R).all() or not np.isfinite(sim_t).all():
        return None
    return sim_scale, sim_R, sim_t


def safe_scalar_float(d: Dict[str, Any], key: str) -> Optional[float]:
    if key not in d:
        return None
    try:
        return float(np.asarray(d[key]).reshape(()))
    except Exception:
        return None


def safe_scalar_int(d: Dict[str, Any], key: str) -> Optional[int]:
    if key not in d:
        return None
    try:
        return int(np.asarray(d[key]).reshape(()))
    except Exception:
        return None


def load_fixed_non_pose_mhr_params(
    npy_dir: Path,
    cam: str,
    include_hand_pose: bool,
) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    npy_path = find_existing_with_exts(npy_dir, cam, NP_EXTS)
    if npy_path is None:
        raise FileNotFoundError(f"Could not find fixed-parameter source file for cam='{cam}' in {npy_dir}")
    d = load_npy_dict(npy_path)
    required = ["scale_params", "shape_params", "expr_params"]
    if include_hand_pose:
        required = ["hand_pose_params"] + required
    missing = [key for key in required if key not in d]
    if missing:
        raise KeyError(
            "Fixed-parameter source is missing required keys: "
            + ", ".join(missing)
            + f" (file: {npy_path})"
        )

    hand: Optional[np.ndarray] = None
    if include_hand_pose:
        hand = np.asarray(d["hand_pose_params"], dtype=np.float32).reshape(-1)
    scale = np.asarray(d["scale_params"], dtype=np.float32).reshape(-1)
    shape = np.asarray(d["shape_params"], dtype=np.float32).reshape(-1)
    expr = np.asarray(d["expr_params"], dtype=np.float32).reshape(-1)

    if hand is not None and not np.isfinite(hand).all():
        raise ValueError(f"Non-finite hand_pose_params in fixed source: {npy_path}")
    if not np.isfinite(scale).all():
        raise ValueError(f"Non-finite scale_params in fixed source: {npy_path}")
    if not np.isfinite(shape).all():
        raise ValueError(f"Non-finite shape_params in fixed source: {npy_path}")
    if not np.isfinite(expr).all():
        raise ValueError(f"Non-finite expr_params in fixed source: {npy_path}")
    return hand, scale, shape, expr


def load_fixed_body_pose_params(
    npy_dir: Path,
    cam: str,
) -> np.ndarray:
    npy_path = find_existing_with_exts(npy_dir, cam, NP_EXTS)
    if npy_path is None:
        raise FileNotFoundError(f"Could not find fixed lower-body source file for cam='{cam}' in {npy_dir}")
    d = load_npy_dict(npy_path)
    if "body_pose_params" not in d:
        raise KeyError(f"Fixed lower-body source is missing 'body_pose_params' (file: {npy_path})")
    body_pose = np.asarray(d["body_pose_params"], dtype=np.float32).reshape(-1)
    if not np.isfinite(body_pose).all():
        raise ValueError(f"Non-finite body_pose_params in fixed lower-body source: {npy_path}")
    return body_pose


def push_temporal_pose_history(
    prev_pose: Optional[np.ndarray],
    prev_prev_pose: Optional[np.ndarray],
    new_pose: np.ndarray,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    updated_prev = np.asarray(new_pose, dtype=np.float32).reshape(-1).copy()
    updated_prev_prev = None if prev_pose is None else np.asarray(prev_pose, dtype=np.float32).reshape(-1).copy()
    return updated_prev, updated_prev_prev


def stale_frame_run_length(frame_results: Sequence[FramePipelineResult]) -> int:
    stale = 0
    for fr in reversed(frame_results):
        if (fr.status == "ok") and (not fr.is_bad_loss):
            break
        stale += 1
    return stale


def recover_missing_and_bad_frames(
    frame_results: List[FramePipelineResult],
    optimization_root: Path,
    optimized_name: str,
    max_edge_copy_span: int,
) -> List[Optional[Dict[str, Any]]]:
    recovery_interp_keys = ("body_pose_params", "hand_pose_params")
    frame_dicts: List[Optional[Dict[str, Any]]] = [None] * len(frame_results)
    valid: List[bool] = [False] * len(frame_results)

    for i, fr in enumerate(frame_results):
        if fr.optimized_npy is None or not fr.optimized_npy.exists():
            continue
        try:
            d = load_npy_dict(fr.optimized_npy)
        except Exception:
            continue
        frame_dicts[i] = d
        valid[i] = (fr.status == "ok") and (not fr.is_bad_loss)

    for i, fr in enumerate(frame_results):
        need_recover = (
            fr.optimized_npy is None
            or frame_dicts[i] is None
            or fr.is_bad_loss
            or fr.status in {
                "missing_input",
                "insufficient_views",
                "triangulation_failed",
                "triangulation_missing",
                "optimization_failed",
                "bad_loss",
            }
        )
        if not need_recover:
            continue

        frame_dicts[i] = None
        valid[i] = False

        prev_i = next((j for j in range(i - 1, -1, -1) if valid[j] and frame_dicts[j] is not None), None)
        next_i = next((j for j in range(i + 1, len(frame_results)) if valid[j] and frame_dicts[j] is not None), None)
        if prev_i is None and next_i is None:
            continue

        if prev_i is not None and next_i is not None and next_i > prev_i:
            prev_dict = frame_dicts[prev_i]
            next_dict = frame_dicts[next_i]
            if prev_dict is None or next_dict is None:
                continue
            alpha = float((i - prev_i) / float(next_i - prev_i))
            recovered = interpolate_frame_dict(prev_dict, next_dict, alpha, keys=recovery_interp_keys)
            fr.status = "recovered_interpolated"
            fr.recovered_from = f"{frame_results[prev_i].rel_dir}->{frame_results[next_i].rel_dir}"
        elif prev_i is not None:
            if int(i - prev_i) > int(max(0, max_edge_copy_span)):
                continue
            prev_dict = frame_dicts[prev_i]
            if prev_dict is None:
                continue
            recovered = copy_frame_dict(prev_dict, mode="copy_prev")
            fr.status = "recovered_copy_prev"
            fr.recovered_from = frame_results[prev_i].rel_dir
        else:
            if next_i is None or int(next_i - i) > int(max(0, max_edge_copy_span)):
                continue
            next_dict = frame_dicts[next_i]
            if next_dict is None:
                continue
            recovered = copy_frame_dict(next_dict, mode="copy_next")
            fr.status = "recovered_copy_next"
            fr.recovered_from = frame_results[next_i].rel_dir

        out_path = fr.optimized_npy if fr.optimized_npy is not None else (optimization_root / fr.rel_dir / optimized_name).resolve()
        if fr.optimized_npy is not None and fr.optimized_npy.exists():
            try:
                fr.optimized_npy.unlink()
            except Exception:
                pass
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for key in (
            "opt_loss_hist",
            "opt_data_loss_hist",
            "opt_best_loss",
            "opt_final_loss",
            "opt_best_data_loss",
            "opt_final_data_loss",
            "opt_best_iter",
            "opt_loss_growth_ratio",
            "opt_data_loss_growth_ratio",
            "opt_used_temporal_init",
            "opt_temporal_weight",
            "opt_temporal_velocity_weight",
            "opt_temporal_accel_weight",
            "opt_temporal_extrapolation",
            "opt_sim_reused_prev",
        ):
            recovered.pop(key, None)
        recovered["opt_recovered_from"] = "" if fr.recovered_from is None else fr.recovered_from

        save_npy_dict(out_path, recovered)
        fr.optimized_npy = out_path
        fr.is_bad_loss = False
        fr.best_loss = None
        fr.final_loss = None
        fr.best_data_loss = None
        fr.final_data_loss = None
        fr.best_iter = None
        frame_dicts[i] = recovered
        valid[i] = True

    return frame_dicts
