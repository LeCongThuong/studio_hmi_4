#!/usr/bin/env python3
"""Unified end-to-end runner for SAM-3D body fitting across all pipeline stages.

Pipeline stages:
1. Stage-1 (`sam3d_inference.py`): infer per-view 2D/3D outputs from images.
2. Stage-2 (`triangulate_mhr3d_gt.py`): triangulate subset 3D GT with robust BA.
3. Stage-3 (`optimize_mhr_pose.py`): optimize MHR pose parameters to triangulated GT.

This runner is sequence-aware for video-like input trees:
  input_frames/<k>/<cam>.jpg  (k = time index)
"""
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sam3d_inference import Demo2Config, run_demo
from optimize_mhr_pose import (
    OptimizationConfig,
    OptimizationRunResult,
    build_optimization_runtime,
    run_optimization,
)
from triangulate_mhr3d_gt import (
    NP_EXTS,
    TriangulationConfig,
    find_existing_with_exts,
    run_triangulation,
)
from video_temporal_utils import (
    copy_frame_dict,
    extract_keypoint_sequence,
    interpolate_frame_dict,
    load_npy_dict,
    save_keypoint_sequence_mp4,
    save_npy_dict,
    show_keypoint_sequence_interactive,
    smooth_frame_dict_sequence,
)


@dataclass
class FullPipelineConfig:
    """Configuration for orchestrating all pipeline stages."""

    image_folder: str
    output_root: str
    cams: List[str]
    caliscope_toml: str
    mhr_py: str = "mhr70.py"
    toml_sections: Optional[List[str]] = None
    checkpoint_path: str = ""
    mhr_path: str = ""
    detector_name: str = "vitdet"
    segmentor_name: str = "sam2"
    fov_name: str = "moge2"
    detector_path: str = ""
    segmentor_path: str = ""
    fov_path: str = ""
    bbox_thresh: float = 0.8
    use_mask: bool = False
    debug_inference: bool = False
    save_mhr_params: bool = False
    person_select_strategy: str = "largest_bbox"
    person_index: int = 0
    frame_rel: Optional[str] = None
    skip_inference: bool = False
    skip_triangulation: bool = False
    skip_optimization: bool = False
    overwrite: bool = False
    npy_root: Optional[str] = None
    triangulated_name: str = "triangulated.npz"
    optimized_name: str = "opt_out.npy"
    normalized: bool = False
    pixel: bool = False
    invert_extrinsics: bool = False
    lm_iters: int = 25
    lm_lambda: float = 1e-3
    lm_eps: float = 1e-4
    score_type: str = "median"
    huber_delta: float = 10.0
    inlier_thresh: float = 30.0
    robust_lm: bool = False
    robust_lm_delta: float = 10.0
    debug_triangulation: bool = False
    debug_triangulation_every_frame: bool = False
    save_triangulation_debug: bool = False
    hf_repo: Optional[str] = None
    opt_ckpt: Optional[str] = None
    opt_mhr_pt: str = ""
    device: str = "cuda"
    iters: int = 200
    lr: float = 5e-2
    with_scale: bool = False
    huber_m: float = 0.03
    w_pose_reg: float = 1e-3
    w_temporal: float = 3e-3
    w_temporal_velocity: float = 2e-3
    w_temporal_accel: float = 5e-4
    temporal_init_blend: float = 0.7
    temporal_extrapolation: float = 1.0
    bad_loss_threshold: float = 3e-5
    bad_data_loss_threshold: float = 2e-5
    bad_loss_growth_ratio: float = 1.5
    min_valid_points: int = 6
    zero_weight_strategy: str = "uniform_finite"
    freeze_lower_body: bool = False
    topk_print: int = 10
    save_opt_debug: bool = False
    bad_frame_max_retries: int = 2
    min_views: int = 2
    recover_bad_frames: bool = True
    fill_missing_frames: bool = True
    max_stale_temporal_frames: int = 40
    max_edge_recovery_copy_span: int = 15
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.65
    smoothing_median_window: int = 5
    smoothing_outlier_sigma: float = 3.5
    smoothed_name: str = "opt_out_smoothed.npy"
    debug_sequence: bool = False
    save_sequence_mp4: bool = False
    sequence_mp4_name: str = "sequence_debug.mp4"
    sequence_fps: int = 20
    save_summary_json: bool = True


@dataclass
class FramePipelineResult:
    """Per-frame summary produced by the full pipeline."""

    rel_dir: str
    frame_index: Optional[int]
    npy_dir: Optional[Path]
    available_cams: List[str]
    used_cams: List[str]
    triangulated_npz: Optional[Path]
    optimized_npy: Optional[Path]
    smoothed_npy: Optional[Path]
    status: str
    best_loss: Optional[float] = None
    final_loss: Optional[float] = None
    best_data_loss: Optional[float] = None
    final_data_loss: Optional[float] = None
    best_iter: Optional[int] = None
    is_bad_loss: bool = False
    recovered_from: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FullPipelineResult:
    """Aggregated output of the full pipeline run."""

    output_root: Path
    npy_root: Path
    frames: List[FramePipelineResult]
    summary_json: Optional[Path] = None


@dataclass
class FrameInputEntry:
    """Input info for a single frame folder."""

    rel_dir: str
    npy_dir: Optional[Path]
    frame_index: Optional[int]
    available_cams: List[str]


def _rel_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if str(rel) == ".":
        return ""
    return rel.as_posix()


def _parse_frame_index(rel_dir: str) -> Optional[int]:
    if rel_dir == "":
        return None
    leaf = rel_dir.replace("\\", "/").split("/")[-1]
    return int(leaf) if leaf.isdigit() else None


def _frame_sort_key(rel_dir: str):
    if rel_dir == "":
        return [(0, "")]
    toks = rel_dir.replace("\\", "/").split("/")
    out = []
    for t in toks:
        if t.isdigit():
            out.append((0, f"{int(t):020d}"))
        else:
            out.append((1, t))
    return out


def _available_cams_in_dir(dir_path: Path, cams: Sequence[str]) -> List[str]:
    return [cam for cam in cams if find_existing_with_exts(dir_path, cam, NP_EXTS) is not None]


def _dir_has_min_cam_predictions(dir_path: Path, cams: Sequence[str], min_views: int) -> bool:
    return len(_available_cams_in_dir(dir_path, cams)) >= int(min_views)


def discover_frame_inputs(
    npy_root: Path,
    cams: Sequence[str],
    frame_rel: Optional[str] = None,
) -> List[FrameInputEntry]:
    """Discover frame directories and available cameras in each frame."""

    if frame_rel:
        target = (npy_root / frame_rel).resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"Requested frame_rel directory not found: {target}")
        rel_dir = _rel_key(target, npy_root)
        return [
            FrameInputEntry(
                rel_dir=rel_dir,
                npy_dir=target,
                frame_index=_parse_frame_index(rel_dir),
                available_cams=_available_cams_in_dir(target, cams),
            )
        ]

    candidates = [npy_root]
    candidates.extend(sorted(p for p in npy_root.rglob("*") if p.is_dir()))

    entries: List[FrameInputEntry] = []
    for p in candidates:
        av = _available_cams_in_dir(p, cams)
        if len(av) == 0:
            continue
        rel_dir = _rel_key(p, npy_root)
        entries.append(
            FrameInputEntry(
                rel_dir=rel_dir,
                npy_dir=p,
                frame_index=_parse_frame_index(rel_dir),
                available_cams=av,
            )
        )

    entries.sort(key=lambda e: _frame_sort_key(e.rel_dir))
    return entries


def _inject_numeric_gaps(entries: List[FrameInputEntry]) -> List[FrameInputEntry]:
    if len(entries) < 2:
        return entries
    if any(e.frame_index is None for e in entries):
        return entries
    if any("/" in e.rel_dir for e in entries if e.rel_dir != ""):
        return entries

    by_idx = {int(e.frame_index): e for e in entries if e.frame_index is not None}
    all_idx = sorted(by_idx.keys())
    out: List[FrameInputEntry] = []
    for idx in range(all_idx[0], all_idx[-1] + 1):
        if idx in by_idx:
            out.append(by_idx[idx])
        else:
            out.append(
                FrameInputEntry(
                    rel_dir=str(idx),
                    npy_dir=None,
                    frame_index=idx,
                    available_cams=[],
                )
            )
    return out


def _cam_to_section_map(cams: Sequence[str], toml_sections: Optional[Sequence[str]]) -> Dict[str, str]:
    if toml_sections is None or len(toml_sections) == 0:
        return {c: c for c in cams}
    if len(toml_sections) != len(cams):
        raise ValueError("--toml_sections must match length of --cams (or omit).")
    return {c: s for c, s in zip(cams, toml_sections)}


def _expected_stage1_meta(config: FullPipelineConfig, image_root: Path) -> Dict[str, Any]:
    include_rel_dirs = [config.frame_rel] if config.frame_rel else None
    return {
        "image_folder": str(image_root.resolve()),
        "include_rel_dirs": include_rel_dirs,
        "checkpoint_path": str(config.checkpoint_path),
        "detector_name": str(config.detector_name),
        "segmentor_name": str(config.segmentor_name),
        "fov_name": str(config.fov_name),
        "person_select_strategy": str(config.person_select_strategy),
        "person_index": int(config.person_index),
    }


def _load_stage1_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stage1_meta_matches(
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
    ]
    for key in keys:
        if existing_meta.get(key) != expected_meta.get(key):
            return False
    return True


def _update_result_from_opt(dst: FramePipelineResult, opt_res: OptimizationRunResult) -> None:
    dst.best_loss = float(opt_res.best_loss)
    dst.final_loss = float(opt_res.final_loss)
    dst.best_data_loss = float(opt_res.best_data_loss)
    dst.final_data_loss = float(opt_res.final_data_loss)
    dst.best_iter = int(opt_res.best_iter)
    dst.is_bad_loss = bool(opt_res.is_bad_loss)


def _load_pose_if_good(opt_npy: Path) -> Optional[np.ndarray]:
    try:
        d = load_npy_dict(opt_npy)
    except Exception:
        return None
    is_bad = bool(int(np.asarray(d.get("opt_is_bad_loss", 0)).reshape(())))
    if is_bad:
        return None
    if "body_pose_params" not in d:
        return None
    return np.asarray(d["body_pose_params"], dtype=np.float32).reshape(-1)


def _load_similarity_if_good(opt_npy: Path) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
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


def _safe_scalar_float(d: Dict[str, Any], key: str) -> Optional[float]:
    if key not in d:
        return None
    try:
        return float(np.asarray(d[key]).reshape(()))
    except Exception:
        return None


def _safe_scalar_int(d: Dict[str, Any], key: str) -> Optional[int]:
    if key not in d:
        return None
    try:
        return int(np.asarray(d[key]).reshape(()))
    except Exception:
        return None


def _push_temporal_pose_history(
    prev_pose: Optional[np.ndarray],
    prev_prev_pose: Optional[np.ndarray],
    new_pose: np.ndarray,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    updated_prev = np.asarray(new_pose, dtype=np.float32).reshape(-1).copy()
    updated_prev_prev = None if prev_pose is None else np.asarray(prev_pose, dtype=np.float32).reshape(-1).copy()
    return updated_prev, updated_prev_prev


def _stale_frame_run_length(frame_results: Sequence[FramePipelineResult]) -> int:
    stale = 0
    for fr in reversed(frame_results):
        if (fr.status in {"ok", "recovered_retry"}) and (not fr.is_bad_loss):
            break
        stale += 1
    return stale


def _recover_missing_and_bad_frames(
    frame_results: List[FramePipelineResult],
    optimization_root: Path,
    optimized_name: str,
    max_edge_copy_span: int,
) -> List[Optional[Dict[str, Any]]]:
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
        valid[i] = (fr.status in {"ok", "recovered_retry"}) and (not fr.is_bad_loss)

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

        # Do not let bad/unusable frame payloads leak into smoothing/debug by default.
        frame_dicts[i] = None
        valid[i] = False

        prev_i = next((j for j in range(i - 1, -1, -1) if valid[j] and frame_dicts[j] is not None), None)
        next_i = next((j for j in range(i + 1, len(frame_results)) if valid[j] and frame_dicts[j] is not None), None)
        if prev_i is None and next_i is None:
            continue

        out_path = (
            fr.optimized_npy
            if fr.optimized_npy is not None
            else (optimization_root / fr.rel_dir / optimized_name).resolve()
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if prev_i is not None and next_i is not None and next_i > prev_i:
            prev_dict = frame_dicts[prev_i]
            next_dict = frame_dicts[next_i]
            if prev_dict is None or next_dict is None:
                continue
            alpha = float((i - prev_i) / float(next_i - prev_i))
            recovered = interpolate_frame_dict(prev_dict, next_dict, alpha)
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
            if next_i is None:
                continue
            if int(next_i - i) > int(max(0, max_edge_copy_span)):
                continue
            next_dict = frame_dicts[next_i]
            if next_dict is None:
                continue
            recovered = copy_frame_dict(next_dict, mode="copy_next")
            fr.status = "recovered_copy_next"
            fr.recovered_from = frame_results[next_i].rel_dir

        # Recovered frames are synthetic; copied optimization metrics from neighbor
        # frames are misleading and must not be interpreted as this frame's optimize stats.
        for k in (
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
            if k in recovered:
                recovered.pop(k, None)
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


def _save_summaries(
    output_root: Path,
    npy_root: Path,
    frame_results: List[FramePipelineResult],
) -> Path:
    status_counts: Dict[str, int] = {}
    for fr in frame_results:
        status_counts[fr.status] = status_counts.get(fr.status, 0) + 1

    payload = {
        "output_root": str(output_root),
        "npy_root": str(npy_root),
        "num_frames": len(frame_results),
        "status_counts": status_counts,
        "frames": [
            {
                "rel_dir": fr.rel_dir,
                "frame_index": fr.frame_index,
                "npy_dir": None if fr.npy_dir is None else str(fr.npy_dir),
                "available_cams": fr.available_cams,
                "used_cams": fr.used_cams,
                "triangulated_npz": None if fr.triangulated_npz is None else str(fr.triangulated_npz),
                "optimized_npy": None if fr.optimized_npy is None else str(fr.optimized_npy),
                "smoothed_npy": None if fr.smoothed_npy is None else str(fr.smoothed_npy),
                "status": fr.status,
                "best_loss": fr.best_loss,
                "final_loss": fr.final_loss,
                "best_data_loss": fr.best_data_loss,
                "final_data_loss": fr.final_data_loss,
                "best_iter": fr.best_iter,
                "is_bad_loss": fr.is_bad_loss,
                "recovered_from": fr.recovered_from,
                "error": fr.error,
            }
            for fr in frame_results
        ],
    }
    out = (output_root / "sequence_summary.json").resolve()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for full pipeline orchestration."""

    ap = argparse.ArgumentParser(
        description="Run SAM -> triangulation -> optimization as one full pipeline."
    )
    ap.add_argument("--image_folder", required=True, type=str, help="Input image root.")
    ap.add_argument("--output_root", required=True, type=str, help="Pipeline output root.")
    ap.add_argument("--cams", nargs="+", required=True, help="Camera names (file stems), e.g. left front right.")
    ap.add_argument("--caliscope_toml", required=True, type=str, help="Path to Caliscope TOML.")
    ap.add_argument("--mhr_py", default="mhr70.py", type=str, help="Path to mhr70.py with pose_info.")
    ap.add_argument(
        "--toml_sections",
        nargs="*",
        default=None,
        help="Optional TOML section names aligned with --cams.",
    )

    # Stage 1: SAM 3D body inference
    ap.add_argument("--checkpoint_path", default="", type=str, help="SAM-3D checkpoint (required unless --skip_inference).")
    ap.add_argument("--mhr_path", default="", type=str, help="MHR model path for inference.")
    ap.add_argument("--detector_name", default="vitdet", type=str)
    ap.add_argument("--segmentor_name", default="sam2", type=str)
    ap.add_argument("--fov_name", default="moge2", type=str)
    ap.add_argument("--detector_path", default="", type=str)
    ap.add_argument("--segmentor_path", default="", type=str)
    ap.add_argument("--fov_path", default="", type=str)
    ap.add_argument("--bbox_thresh", default=0.8, type=float)
    ap.add_argument("--use_mask", action="store_true", default=False)
    ap.add_argument("--debug_inference", action="store_true", default=False)
    ap.add_argument("--save_mhr_params", action="store_true", default=False)
    ap.add_argument(
        "--person_select_strategy",
        type=str,
        default="largest_bbox",
        choices=["first", "largest_bbox", "person_index"],
        help="How stage-1 selects one person if multiple are detected.",
    )
    ap.add_argument(
        "--person_index",
        type=int,
        default=0,
        help="Person index used when --person_select_strategy=person_index.",
    )

    # Pipeline control
    ap.add_argument("--frame_rel", default=None, type=str, help="Optional relative frame dir under npy root.")
    ap.add_argument("--skip_inference", action="store_true", default=False)
    ap.add_argument("--skip_triangulation", action="store_true", default=False)
    ap.add_argument("--skip_optimization", action="store_true", default=False)
    ap.add_argument("--overwrite", action="store_true", default=False, help="Recompute outputs even when they already exist.")
    ap.add_argument("--npy_root", default=None, type=str, help="Existing npy root (used when --skip_inference).")
    ap.add_argument("--triangulated_name", default="triangulated.npz", type=str)
    ap.add_argument("--optimized_name", default="opt_out.npy", type=str)
    ap.add_argument("--min_views", type=int, default=2, help="Minimum available views required per frame.")
    ap.add_argument("--no_recover_bad_frames", action="store_true", help="Disable recovery for bad/failed frames.")
    ap.add_argument("--no_fill_missing_frames", action="store_true", help="Disable insertion of missing numeric frame folders.")
    ap.add_argument(
        "--max_stale_temporal_frames",
        type=int,
        default=40,
        help="Disable temporal init/priors after this many consecutive non-good frames (0 disables this safeguard).",
    )
    ap.add_argument(
        "--max_edge_recovery_copy_span",
        type=int,
        default=15,
        help="Max frame distance for one-sided recovery copy (copy_prev/copy_next). Larger gaps stay unrecovered.",
    )

    # Stage 2: Triangulation + BA
    ap.add_argument("--normalized", action="store_true", default=False)
    ap.add_argument("--pixel", action="store_true", default=False)
    ap.add_argument("--invert_extrinsics", action="store_true", default=False)
    ap.add_argument("--lm_iters", type=int, default=25)
    ap.add_argument("--lm_lambda", type=float, default=1e-3)
    ap.add_argument("--lm_eps", type=float, default=1e-4)
    ap.add_argument("--score_type", type=str, default="median", choices=["median", "trimmed", "huber"])
    ap.add_argument("--huber_delta", type=float, default=10.0)
    ap.add_argument("--inlier_thresh", type=float, default=30.0)
    ap.add_argument("--robust_lm", action="store_true", default=False)
    ap.add_argument("--robust_lm_delta", type=float, default=10.0)
    ap.add_argument("--debug_triangulation", action="store_true", default=False, help="Interactive 3D display.")
    ap.add_argument(
        "--debug_triangulation_every_frame",
        action="store_true",
        default=False,
        help="When --debug_triangulation is enabled, show interactive 3D window for every frame (default: first frame only).",
    )
    ap.add_argument("--save_triangulation_debug", action="store_true", default=False, help="Save per-frame overlay debug.")

    # Stage 3: Optimization
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--hf_repo", type=str, default=None)
    grp.add_argument("--opt_ckpt", type=str, default=None)
    ap.add_argument("--opt_mhr_pt", type=str, default="")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--with_scale", action="store_true")
    ap.add_argument("--huber_m", type=float, default=0.03)
    ap.add_argument("--w_pose_reg", type=float, default=1e-3)
    ap.add_argument("--w_temporal", type=float, default=3e-3)
    ap.add_argument("--w_temporal_velocity", type=float, default=2e-3)
    ap.add_argument("--w_temporal_accel", type=float, default=5e-4)
    ap.add_argument("--temporal_init_blend", type=float, default=0.7)
    ap.add_argument("--temporal_extrapolation", type=float, default=1.0)
    ap.add_argument("--bad_loss_threshold", type=float, default=3e-5)
    ap.add_argument("--bad_data_loss_threshold", type=float, default=2e-5)
    ap.add_argument("--bad_loss_growth_ratio", type=float, default=1.5)
    ap.add_argument("--min_valid_points", type=int, default=6)
    ap.add_argument(
        "--zero_weight_strategy",
        type=str,
        choices=["uniform_finite", "fail"],
        default="uniform_finite",
    )
    ap.add_argument(
        "--freeze_lower_body",
        action="store_true",
        help="Freeze lower-body pose dimensions in stage-3 optimization.",
    )
    ap.add_argument("--bad_frame_max_retries", type=int, default=2)
    ap.add_argument("--topk_print", type=int, default=10)
    ap.add_argument("--save_opt_debug", action="store_true", default=False, help="Save optimization debug plots/artifacts.")

    # Sequence post-process and debug
    ap.add_argument("--disable_smoothing", action="store_true", help="Disable temporal smoothing module.")
    ap.add_argument("--smoothing_alpha", type=float, default=0.65)
    ap.add_argument("--smoothing_median_window", type=int, default=5)
    ap.add_argument("--smoothing_outlier_sigma", type=float, default=3.5)
    ap.add_argument("--smoothed_name", type=str, default="opt_out_smoothed.npy")
    ap.add_argument("--debug_sequence", action="store_true", help="Show interactive 4D keypoint motion viewer.")
    ap.add_argument("--debug_4d", action="store_true", help="Alias of --debug_sequence.")
    ap.add_argument("--save_sequence_mp4", action="store_true", help="Export sequence debug MP4.")
    ap.add_argument("--save_4d_mp4", action="store_true", help="Alias of --save_sequence_mp4.")
    ap.add_argument("--sequence_mp4_name", type=str, default="sequence_debug.mp4")
    ap.add_argument("--sequence_fps", type=int, default=20)
    ap.add_argument("--no_summary_json", action="store_true", help="Disable writing sequence_summary.json.")
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args for full pipeline execution."""

    parser = build_arg_parser()
    return parser.parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> FullPipelineConfig:
    """Convert parsed CLI args to `FullPipelineConfig`."""

    image_folder = getattr(args, "image_folder", None)
    output_root = getattr(args, "output_root", None)
    cams = getattr(args, "cams", None)
    caliscope_toml = getattr(args, "caliscope_toml", None)
    missing_required = [
        name
        for name, value in (
            ("image_folder", image_folder),
            ("output_root", output_root),
            ("cams", cams),
            ("caliscope_toml", caliscope_toml),
        )
        if value is None
    ]
    if missing_required:
        raise AttributeError(
            "Missing required args for namespace_to_config: "
            + ", ".join(missing_required)
        )
    if not isinstance(cams, (list, tuple)):
        raise AttributeError("Argument --cams must be a sequence of camera names.")

    debug_sequence = bool(
        getattr(args, "debug_sequence", False) or getattr(args, "debug_4d", False)
    )
    save_sequence_mp4 = bool(
        getattr(args, "save_sequence_mp4", False) or getattr(args, "save_4d_mp4", False)
    )

    return FullPipelineConfig(
        image_folder=str(image_folder),
        output_root=str(output_root),
        cams=list(cams),
        caliscope_toml=str(caliscope_toml),
        mhr_py=getattr(args, "mhr_py", "mhr70.py"),
        toml_sections=getattr(args, "toml_sections", None),
        checkpoint_path=getattr(args, "checkpoint_path", ""),
        mhr_path=getattr(args, "mhr_path", ""),
        detector_name=getattr(args, "detector_name", "vitdet"),
        segmentor_name=getattr(args, "segmentor_name", "sam2"),
        fov_name=getattr(args, "fov_name", "moge2"),
        detector_path=getattr(args, "detector_path", ""),
        segmentor_path=getattr(args, "segmentor_path", ""),
        fov_path=getattr(args, "fov_path", ""),
        bbox_thresh=float(getattr(args, "bbox_thresh", 0.8)),
        use_mask=bool(getattr(args, "use_mask", False)),
        debug_inference=bool(getattr(args, "debug_inference", False)),
        save_mhr_params=bool(getattr(args, "save_mhr_params", False)),
        person_select_strategy=getattr(args, "person_select_strategy", "largest_bbox"),
        person_index=int(getattr(args, "person_index", 0)),
        frame_rel=getattr(args, "frame_rel", None),
        skip_inference=bool(getattr(args, "skip_inference", False)),
        skip_triangulation=bool(getattr(args, "skip_triangulation", False)),
        skip_optimization=bool(getattr(args, "skip_optimization", False)),
        overwrite=bool(getattr(args, "overwrite", False)),
        npy_root=getattr(args, "npy_root", None),
        triangulated_name=getattr(args, "triangulated_name", "triangulated.npz"),
        optimized_name=getattr(args, "optimized_name", "opt_out.npy"),
        normalized=bool(getattr(args, "normalized", False)),
        pixel=bool(getattr(args, "pixel", False)),
        invert_extrinsics=bool(getattr(args, "invert_extrinsics", False)),
        lm_iters=int(getattr(args, "lm_iters", 25)),
        lm_lambda=float(getattr(args, "lm_lambda", 1e-3)),
        lm_eps=float(getattr(args, "lm_eps", 1e-4)),
        score_type=getattr(args, "score_type", "median"),
        huber_delta=float(getattr(args, "huber_delta", 10.0)),
        inlier_thresh=float(getattr(args, "inlier_thresh", 30.0)),
        robust_lm=bool(getattr(args, "robust_lm", False)),
        robust_lm_delta=float(getattr(args, "robust_lm_delta", 10.0)),
        debug_triangulation=bool(getattr(args, "debug_triangulation", False)),
        debug_triangulation_every_frame=bool(getattr(args, "debug_triangulation_every_frame", False)),
        save_triangulation_debug=bool(getattr(args, "save_triangulation_debug", False)),
        hf_repo=getattr(args, "hf_repo", None),
        opt_ckpt=getattr(args, "opt_ckpt", None),
        opt_mhr_pt=getattr(args, "opt_mhr_pt", ""),
        device=getattr(args, "device", "cuda"),
        iters=int(getattr(args, "iters", 200)),
        lr=float(getattr(args, "lr", 5e-2)),
        with_scale=bool(getattr(args, "with_scale", False)),
        huber_m=float(getattr(args, "huber_m", 0.03)),
        w_pose_reg=float(getattr(args, "w_pose_reg", 1e-3)),
        w_temporal=float(getattr(args, "w_temporal", 3e-3)),
        w_temporal_velocity=float(getattr(args, "w_temporal_velocity", 2e-3)),
        w_temporal_accel=float(getattr(args, "w_temporal_accel", 5e-4)),
        temporal_init_blend=float(getattr(args, "temporal_init_blend", 0.7)),
        temporal_extrapolation=float(getattr(args, "temporal_extrapolation", 1.0)),
        bad_loss_threshold=float(getattr(args, "bad_loss_threshold", 3e-5)),
        bad_data_loss_threshold=float(getattr(args, "bad_data_loss_threshold", 2e-5)),
        bad_loss_growth_ratio=float(getattr(args, "bad_loss_growth_ratio", 1.5)),
        min_valid_points=int(getattr(args, "min_valid_points", 6)),
        zero_weight_strategy=getattr(args, "zero_weight_strategy", "uniform_finite"),
        freeze_lower_body=bool(getattr(args, "freeze_lower_body", False)),
        topk_print=int(getattr(args, "topk_print", 10)),
        save_opt_debug=bool(getattr(args, "save_opt_debug", False)),
        bad_frame_max_retries=int(getattr(args, "bad_frame_max_retries", 2)),
        min_views=int(getattr(args, "min_views", 2)),
        recover_bad_frames=not bool(getattr(args, "no_recover_bad_frames", False)),
        fill_missing_frames=not bool(getattr(args, "no_fill_missing_frames", False)),
        max_stale_temporal_frames=int(getattr(args, "max_stale_temporal_frames", 40)),
        max_edge_recovery_copy_span=int(getattr(args, "max_edge_recovery_copy_span", 15)),
        enable_smoothing=not bool(getattr(args, "disable_smoothing", False)),
        smoothing_alpha=float(getattr(args, "smoothing_alpha", 0.65)),
        smoothing_median_window=int(getattr(args, "smoothing_median_window", 5)),
        smoothing_outlier_sigma=float(getattr(args, "smoothing_outlier_sigma", 3.5)),
        smoothed_name=getattr(args, "smoothed_name", "opt_out_smoothed.npy"),
        debug_sequence=debug_sequence,
        save_sequence_mp4=save_sequence_mp4,
        sequence_mp4_name=getattr(args, "sequence_mp4_name", "sequence_debug.mp4"),
        sequence_fps=int(getattr(args, "sequence_fps", 20)),
        save_summary_json=not bool(getattr(args, "no_summary_json", False)),
    )


def run_full_pipeline(config: FullPipelineConfig) -> FullPipelineResult:
    """Run all enabled stages and return per-frame outputs."""

    image_root = Path(config.image_folder).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    inference_root = output_root / "inference"
    triangulation_root = output_root / "triangulation"
    optimization_root = output_root / "optimization"
    output_root.mkdir(parents=True, exist_ok=True)

    if not config.skip_inference and not config.checkpoint_path:
        raise ValueError("--checkpoint_path is required unless --skip_inference is set.")

    min_views = max(2, int(config.min_views))
    cam_to_section = _cam_to_section_map(config.cams, config.toml_sections)

    opt_ckpt = config.opt_ckpt or (config.checkpoint_path if not config.hf_repo else None)
    opt_mhr_pt = config.opt_mhr_pt or config.mhr_path
    if not config.skip_optimization and not (config.hf_repo or opt_ckpt):
        raise ValueError("Optimization requires --hf_repo or --opt_ckpt.")

    if config.skip_inference:
        npy_root = (
            Path(config.npy_root).expanduser().resolve()
            if config.npy_root
            else (inference_root / "npy").resolve()
        )
        if not npy_root.is_dir():
            raise FileNotFoundError(f"Numpy root not found: {npy_root}")
    else:
        inferred_npy_root = (inference_root / "npy").resolve()
        stage1_meta_path = (inference_root / "stage1_meta.json").resolve()
        expected_stage1_meta = _expected_stage1_meta(config=config, image_root=image_root)
        reuse_stage1 = False
        if (
            not config.overwrite
            and config.frame_rel is not None
            and inferred_npy_root.is_dir()
        ):
            rel_dir_path = (inferred_npy_root / config.frame_rel).resolve()
            meta_matches = _stage1_meta_matches(
                existing_meta=_load_stage1_meta(stage1_meta_path),
                expected_meta=expected_stage1_meta,
            )
            if (
                rel_dir_path.is_dir()
                and _dir_has_min_cam_predictions(rel_dir_path, config.cams, min_views=min_views)
                and meta_matches
            ):
                reuse_stage1 = True

        if reuse_stage1:
            npy_root = inferred_npy_root
            print(f"[PIPELINE] Reusing existing stage-1 outputs at: {npy_root}")
        else:
            demo_cfg = Demo2Config(
                image_folder=str(image_root),
                output_folder=str(inference_root),
                checkpoint_path=config.checkpoint_path,
                detector_name=config.detector_name,
                segmentor_name=config.segmentor_name,
                fov_name=config.fov_name,
                detector_path=config.detector_path,
                segmentor_path=config.segmentor_path,
                fov_path=config.fov_path,
                mhr_path=config.mhr_path,
                bbox_thresh=config.bbox_thresh,
                use_mask=config.use_mask,
                debug=config.debug_inference,
                save_mhr_params=config.save_mhr_params,
                include_rel_dirs=[config.frame_rel] if config.frame_rel else None,
                person_select_strategy=config.person_select_strategy,
                person_index=config.person_index,
            )
            demo_result = run_demo(demo_cfg)
            npy_root = demo_result.npy_root.resolve()

    frame_inputs = discover_frame_inputs(
        npy_root=npy_root,
        cams=config.cams,
        frame_rel=config.frame_rel,
    )
    if config.fill_missing_frames and config.frame_rel is None:
        frame_inputs = _inject_numeric_gaps(frame_inputs)

    if not frame_inputs:
        raise FileNotFoundError(
            f"No frame directories with camera predictions {config.cams} found under {npy_root}"
        )

    total_frames = len(frame_inputs)
    tri_debug_shown_once = False
    if config.debug_triangulation and (not config.debug_triangulation_every_frame) and total_frames > 1:
        print(
            "[PIPELINE] --debug_triangulation enabled for sequence: "
            "interactive 3D debug will open only for the first processed frame. "
            "Use --debug_triangulation_every_frame to force every frame."
        )

    opt_runtime = None
    if not config.skip_optimization:
        runtime_seed = next(
            (e for e in frame_inputs if e.npy_dir is not None and len(e.available_cams) >= min_views),
            None,
        )
        if runtime_seed is None:
            raise FileNotFoundError(
                f"No frame has enough views (>= {min_views}) for optimization runtime initialization."
            )
        runtime_seed_npy_dir = runtime_seed.npy_dir
        if runtime_seed_npy_dir is None:
            raise FileNotFoundError(
                "Optimization runtime initialization failed: runtime seed has no npy_dir."
            )
        runtime_cfg = OptimizationConfig(
            npz=Path("runtime.npz"),
            npy_dir=runtime_seed_npy_dir,
            cams=list(runtime_seed.available_cams),
            out_npy=Path("runtime.npy"),
            debug_dir=(optimization_root / "debug_opt"),
            hf_repo=config.hf_repo,
            ckpt=opt_ckpt,
            mhr_pt=opt_mhr_pt,
            device=config.device,
            save_debug_artifacts=False,
            min_valid_points=config.min_valid_points,
            zero_weight_strategy=config.zero_weight_strategy,
            freeze_lower_body=config.freeze_lower_body,
        )
        opt_runtime = build_optimization_runtime(runtime_cfg)

    prev_good_pose: Optional[np.ndarray] = None
    prev_prev_good_pose: Optional[np.ndarray] = None
    prev_good_sim_scale: Optional[float] = None
    prev_good_sim_R: Optional[np.ndarray] = None
    prev_good_sim_t: Optional[np.ndarray] = None
    frame_results: List[FramePipelineResult] = []
    for idx, entry in enumerate(frame_inputs, start=1):
        rel_dir = entry.rel_dir
        print(f"[PIPELINE] Frame {idx}/{len(frame_inputs)} rel='{rel_dir or '.'}'")

        fr = FramePipelineResult(
            rel_dir=rel_dir,
            frame_index=entry.frame_index,
            npy_dir=entry.npy_dir,
            available_cams=list(entry.available_cams),
            used_cams=[],
            triangulated_npz=None,
            optimized_npy=None,
            smoothed_npy=None,
            status="pending",
        )

        if entry.npy_dir is None:
            fr.status = "missing_input"
            frame_results.append(fr)
            continue

        used_cams = list(entry.available_cams)
        fr.used_cams = used_cams
        if len(used_cams) < min_views:
            fr.status = "insufficient_views"
            frame_results.append(fr)
            continue

        tri_out = (triangulation_root / rel_dir / config.triangulated_name).resolve()
        fr.triangulated_npz = tri_out
        tri_out.parent.mkdir(parents=True, exist_ok=True)
        tri_debug_dir = (tri_out.parent / "debug") if config.save_triangulation_debug else None

        rel_img_dir = image_root / rel_dir
        img_dir = rel_img_dir if rel_img_dir.is_dir() else image_root

        if not config.skip_triangulation:
            if tri_out.exists() and not config.overwrite:
                print(f"[PIPELINE] Reusing triangulation: {tri_out}")
            else:
                try:
                    tri_debug_enabled = bool(
                        config.debug_triangulation
                        and (
                            config.debug_triangulation_every_frame
                            or (not tri_debug_shown_once)
                        )
                    )
                    tri_cfg = TriangulationConfig(
                        mhr_py=config.mhr_py,
                        caliscope_toml=config.caliscope_toml,
                        cams=used_cams,
                        toml_sections=[cam_to_section[c] for c in used_cams],
                        npy_dir=str(entry.npy_dir),
                        out_npz=str(tri_out),
                        normalized=config.normalized,
                        pixel=config.pixel,
                        invert_extrinsics=config.invert_extrinsics,
                        lm_iters=config.lm_iters,
                        lm_lambda=config.lm_lambda,
                        lm_eps=config.lm_eps,
                        debug=tri_debug_enabled,
                        debug_dir=str(tri_debug_dir) if tri_debug_dir else None,
                        img_dir=str(img_dir),
                        score_type=config.score_type,
                        huber_delta=config.huber_delta,
                        inlier_thresh=config.inlier_thresh,
                        robust_lm=config.robust_lm,
                        robust_lm_delta=config.robust_lm_delta,
                    )
                    run_triangulation(tri_cfg)
                    if tri_debug_enabled and (not config.debug_triangulation_every_frame):
                        tri_debug_shown_once = True
                except Exception as exc:
                    fr.status = "triangulation_failed"
                    fr.error = f"{type(exc).__name__}: {exc}"
                    print(f"[PIPELINE][WARN] Triangulation failed at '{rel_dir}': {fr.error}")
                    traceback.print_exc()
                    frame_results.append(fr)
                    continue
        elif not tri_out.exists():
            fr.status = "triangulation_missing"
            frame_results.append(fr)
            continue

        if config.skip_optimization:
            fr.status = "triangulated"
            frame_results.append(fr)
            continue

        optimized_npy = (optimization_root / rel_dir / config.optimized_name).resolve()
        fr.optimized_npy = optimized_npy

        if optimized_npy.exists() and not config.overwrite:
            print(f"[PIPELINE] Reusing optimized output: {optimized_npy}")
            try:
                d = load_npy_dict(optimized_npy)
                is_bad = bool(int(np.asarray(d.get("opt_is_bad_loss", 0)).reshape(())))
                is_recovered = bool(int(np.asarray(d.get("opt_recovered", 0)).reshape(())))

                pose = None
                if (not is_bad) and ("body_pose_params" in d):
                    try:
                        pose = np.asarray(d["body_pose_params"], dtype=np.float32).reshape(-1)
                    except Exception:
                        pose = None

                if pose is not None:
                    if is_recovered:
                        mode = str(d.get("opt_recovery_mode", "")).strip()
                        if mode == "copy_prev":
                            fr.status = "recovered_copy_prev"
                        elif mode == "copy_next":
                            fr.status = "recovered_copy_next"
                        elif mode == "interpolate":
                            fr.status = "recovered_interpolated"
                        else:
                            fr.status = "recovered"
                        rec_from = str(d.get("opt_recovered_from", "")).strip()
                        fr.recovered_from = rec_from or None
                    else:
                        fr.status = "ok"
                        prev_good_pose, prev_prev_good_pose = _push_temporal_pose_history(
                            prev_pose=prev_good_pose,
                            prev_prev_pose=prev_prev_good_pose,
                            new_pose=pose,
                        )
                        sim = _load_similarity_if_good(optimized_npy)
                        if sim is not None:
                            prev_good_sim_scale = float(sim[0])
                            prev_good_sim_R = sim[1].copy()
                            prev_good_sim_t = sim[2].copy()
                else:
                    fr.status = "bad_loss"
                    fr.is_bad_loss = True

                # Recovered artifacts intentionally do not carry valid per-frame optimization metrics.
                if is_recovered:
                    fr.best_loss = None
                    fr.final_loss = None
                    fr.best_data_loss = None
                    fr.final_data_loss = None
                    fr.best_iter = None
                    fr.is_bad_loss = False
                else:
                    fr.best_loss = _safe_scalar_float(d, "opt_best_loss")
                    fr.final_loss = _safe_scalar_float(d, "opt_final_loss")
                    fr.best_data_loss = _safe_scalar_float(d, "opt_best_data_loss")
                    fr.final_data_loss = _safe_scalar_float(d, "opt_final_data_loss")
                    fr.best_iter = _safe_scalar_int(d, "opt_best_iter")
                    if "opt_is_bad_loss" in d:
                        fr.is_bad_loss = bool(int(np.asarray(d["opt_is_bad_loss"]).reshape(())))
                        if fr.is_bad_loss:
                            fr.status = "bad_loss"
            except Exception:
                pass
            frame_results.append(fr)
            continue

        stale_run = _stale_frame_run_length(frame_results)
        temporal_guard_enabled = int(config.max_stale_temporal_frames) > 0
        use_temporal_priors = not (
            temporal_guard_enabled and stale_run >= int(config.max_stale_temporal_frames)
        )
        if not use_temporal_priors:
            print(
                f"[PIPELINE] Temporal priors disabled at '{rel_dir}' after "
                f"{stale_run} consecutive non-good frames."
            )

        init_prev_body_pose = None if (not use_temporal_priors or prev_good_pose is None) else prev_good_pose.copy()
        init_prev_prev_body_pose = (
            None
            if (not use_temporal_priors or prev_prev_good_pose is None)
            else prev_prev_good_pose.copy()
        )
        init_prev_sim_scale = None if (not use_temporal_priors) else prev_good_sim_scale
        init_prev_sim_R = None if (not use_temporal_priors or prev_good_sim_R is None) else prev_good_sim_R.copy()
        init_prev_sim_t = None if (not use_temporal_priors or prev_good_sim_t is None) else prev_good_sim_t.copy()

        try:
            opt_cfg = OptimizationConfig(
                npz=tri_out,
                npy_dir=entry.npy_dir,
                cams=used_cams,
                out_npy=optimized_npy,
                debug_dir=(optimization_root / rel_dir / "debug_opt").resolve(),
                hf_repo=config.hf_repo,
                ckpt=opt_ckpt,
                mhr_pt=opt_mhr_pt,
                device=config.device,
                iters=config.iters,
                lr=config.lr,
                with_scale=config.with_scale,
                huber_m=config.huber_m,
                w_pose_reg=config.w_pose_reg,
                w_temporal=config.w_temporal,
                w_temporal_velocity=config.w_temporal_velocity,
                w_temporal_accel=config.w_temporal_accel,
                temporal_init_blend=config.temporal_init_blend,
                temporal_extrapolation=config.temporal_extrapolation,
                init_prev_body_pose=init_prev_body_pose,
                init_prev_prev_body_pose=init_prev_prev_body_pose,
                init_prev_sim_scale=init_prev_sim_scale,
                init_prev_sim_R=init_prev_sim_R,
                init_prev_sim_t=init_prev_sim_t,
                bad_loss_threshold=config.bad_loss_threshold,
                bad_data_loss_threshold=config.bad_data_loss_threshold,
                bad_loss_growth_ratio=config.bad_loss_growth_ratio,
                min_valid_points=config.min_valid_points,
                zero_weight_strategy=config.zero_weight_strategy,
                freeze_lower_body=config.freeze_lower_body,
                topk_print=config.topk_print,
                save_debug_artifacts=config.save_opt_debug,
            )
            opt_res = run_optimization(opt_cfg, runtime=opt_runtime)
            _update_result_from_opt(fr, opt_res)
            fr.status = "bad_loss" if opt_res.is_bad_loss else "ok"
            fr.is_bad_loss = bool(opt_res.is_bad_loss)
            if not opt_res.is_bad_loss:
                prev_good_pose, prev_prev_good_pose = _push_temporal_pose_history(
                    prev_pose=prev_good_pose,
                    prev_prev_pose=prev_prev_good_pose,
                    new_pose=opt_res.best_pose,
                )
                prev_good_sim_scale = float(opt_res.sim_scale)
                prev_good_sim_R = np.asarray(opt_res.sim_R, dtype=np.float32).copy()
                prev_good_sim_t = np.asarray(opt_res.sim_t, dtype=np.float32).copy()
        except Exception as exc:
            fr.status = "optimization_failed"
            fr.error = f"{type(exc).__name__}: {exc}"
            print(f"[PIPELINE][WARN] Optimization failed at '{rel_dir}': {fr.error}")
            traceback.print_exc()
            frame_results.append(fr)
            continue

        if (
            fr.is_bad_loss
            and config.recover_bad_frames
            and prev_good_pose is not None
            and use_temporal_priors
        ):
            print(f"[PIPELINE] Retrying bad-loss frame '{rel_dir}' with stronger temporal prior")
            max_retries = max(1, int(config.bad_frame_max_retries))
            for retry_i in range(max_retries):
                gain = float(1.0 + (retry_i + 1) * 2.0)
                try:
                    retry_cfg = OptimizationConfig(
                        npz=tri_out,
                        npy_dir=entry.npy_dir,
                        cams=used_cams,
                        out_npy=optimized_npy,
                        debug_dir=(optimization_root / rel_dir / f"debug_opt_retry_{retry_i + 1}").resolve(),
                        hf_repo=config.hf_repo,
                        ckpt=opt_ckpt,
                        mhr_pt=opt_mhr_pt,
                        device=config.device,
                        iters=max(80, int(config.iters * 0.8)),
                        lr=max(7e-4, config.lr * (0.5 ** (retry_i + 1))),
                        with_scale=config.with_scale,
                        huber_m=config.huber_m,
                        w_pose_reg=config.w_pose_reg,
                        w_temporal=max(config.w_temporal * gain, 3e-4),
                        w_temporal_velocity=max(config.w_temporal_velocity * gain, 2e-4),
                        w_temporal_accel=max(config.w_temporal_accel * gain, 1e-4),
                        temporal_init_blend=max(config.temporal_init_blend, 0.9),
                        temporal_extrapolation=max(config.temporal_extrapolation, 1.0),
                        init_prev_body_pose=prev_good_pose.copy(),
                        init_prev_prev_body_pose=None if prev_prev_good_pose is None else prev_prev_good_pose.copy(),
                        init_prev_sim_scale=prev_good_sim_scale,
                        init_prev_sim_R=None if prev_good_sim_R is None else prev_good_sim_R.copy(),
                        init_prev_sim_t=None if prev_good_sim_t is None else prev_good_sim_t.copy(),
                        bad_loss_threshold=config.bad_loss_threshold,
                        bad_data_loss_threshold=config.bad_data_loss_threshold,
                        bad_loss_growth_ratio=config.bad_loss_growth_ratio,
                        min_valid_points=config.min_valid_points,
                        zero_weight_strategy=config.zero_weight_strategy,
                        freeze_lower_body=config.freeze_lower_body,
                        topk_print=config.topk_print,
                        save_debug_artifacts=config.save_opt_debug,
                    )
                    retry_res = run_optimization(retry_cfg, runtime=opt_runtime)
                    improved_total = retry_res.best_loss < (fr.best_loss or float("inf"))
                    improved_data = (
                        fr.best_data_loss is None
                        or retry_res.best_data_loss < float(fr.best_data_loss)
                    )
                    if (not retry_res.is_bad_loss) or improved_total or improved_data:
                        _update_result_from_opt(fr, retry_res)
                        fr.is_bad_loss = bool(retry_res.is_bad_loss)
                        fr.status = "recovered_retry" if not retry_res.is_bad_loss else "bad_loss"
                        if not retry_res.is_bad_loss:
                            prev_good_pose, prev_prev_good_pose = _push_temporal_pose_history(
                                prev_pose=prev_good_pose,
                                prev_prev_pose=prev_prev_good_pose,
                                new_pose=retry_res.best_pose,
                            )
                            prev_good_sim_scale = float(retry_res.sim_scale)
                            prev_good_sim_R = np.asarray(retry_res.sim_R, dtype=np.float32).copy()
                            prev_good_sim_t = np.asarray(retry_res.sim_t, dtype=np.float32).copy()
                            break
                except Exception as exc:
                    print(
                        f"[PIPELINE][WARN] Retry {retry_i + 1}/{max_retries} failed at "
                        f"'{rel_dir}': {type(exc).__name__}: {exc}"
                    )

        frame_results.append(fr)

    smoothed_dicts: Optional[List[Optional[Dict[str, Any]]]] = None
    if not config.skip_optimization:
        frame_dicts = _recover_missing_and_bad_frames(
            frame_results=frame_results,
            optimization_root=optimization_root,
            optimized_name=config.optimized_name,
            max_edge_copy_span=config.max_edge_recovery_copy_span,
        ) if config.recover_bad_frames else [
            (load_npy_dict(fr.optimized_npy) if fr.optimized_npy is not None and fr.optimized_npy.exists() else None)
            for fr in frame_results
        ]

        if config.enable_smoothing:
            smoothed_dicts = smooth_frame_dict_sequence(
                frame_dicts,
                alpha=config.smoothing_alpha,
                median_window=config.smoothing_median_window,
                outlier_sigma=config.smoothing_outlier_sigma,
            )
            for fr, smoothed_entry in zip(frame_results, smoothed_dicts):
                if smoothed_entry is None:
                    continue
                smoothed_out = (optimization_root / fr.rel_dir / config.smoothed_name).resolve()
                save_npy_dict(smoothed_out, smoothed_entry)
                fr.smoothed_npy = smoothed_out

        if config.debug_sequence or config.save_sequence_mp4:
            seq_source = smoothed_dicts if smoothed_dicts is not None else frame_dicts
            points_seq = extract_keypoint_sequence(seq_source)
            if points_seq.shape[0] > 0:
                if config.save_sequence_mp4:
                    mp4_path = (output_root / config.sequence_mp4_name).resolve()
                    ok = save_keypoint_sequence_mp4(
                        points_seq,
                        mp4_path,
                        fps=config.sequence_fps,
                        title="4D Reconstruction",
                    )
                    if ok:
                        print(f"[PIPELINE] Saved sequence debug mp4: {mp4_path}")
                    else:
                        print("[PIPELINE][WARN] Failed to save sequence debug mp4.")
                if config.debug_sequence:
                    show_keypoint_sequence_interactive(points_seq, title="4D Reconstruction")

    summary_json = None
    if config.save_summary_json:
        summary_json = _save_summaries(
            output_root=output_root,
            npy_root=npy_root,
            frame_results=frame_results,
        )
        print(f"[PIPELINE] Saved summary: {summary_json}")

    return FullPipelineResult(
        output_root=output_root,
        npy_root=npy_root,
        frames=frame_results,
        summary_json=summary_json,
    )


def main(args: Optional[argparse.Namespace] = None) -> FullPipelineResult:
    """CLI/programmatic entrypoint for full pipeline execution."""

    if args is None:
        args = parse_args()
    config = namespace_to_config(args)
    return run_full_pipeline(config)


if __name__ == "__main__":
    main()
