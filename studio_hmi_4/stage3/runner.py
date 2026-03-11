#!/usr/bin/env python3
"""Thin CLI/API surface for stage-3 optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .alignment import (
    MHR_PARAM_HAND_IDXS_133,
    ALIGNMENT_ANCHOR_NAMES,
    build_alignment_anchor_local_indices,
    build_base_keep_mask,
    classify_bad_optimization,
    huber,
    resolve_lower_body_pose_indices,
    resolve_valid_indices_for_prediction,
    safe_growth,
    sanitize_subset_and_weights,
    select_alignment_subset_tensors,
    umeyama_similarity,
)
from .debug import plot_3d_compare, plot_loss_curve, set_axes_equal
from .pipeline import run_optimization
from .runtime import (
    DEFAULT_PARAM_DIMS,
    apply_repo_camera_flip_xyz,
    build_optimization_runtime,
    ensure_dir,
    find_npy_for_cam,
    get_param_array,
    load_body_pose_from_npy,
    load_sam_head,
    mhr_fk,
    safe_numpy_item_load,
    to_torch,
)
from .types import OptimizationConfig, OptimizationRunResult, OptimizationRuntime


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--npy_dir", required=True, type=Path)
    ap.add_argument("--cams", nargs="+", required=True, help="Camera names, e.g. left front right")
    ap.add_argument("--out_npy", required=True, type=Path)
    ap.add_argument("--debug_dir", default=Path("debug_opt"), type=Path)

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hf_repo", type=str, default=None)
    grp.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mhr_pt", type=str, default="", help="needed if using --ckpt")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--with_scale", action="store_true")
    ap.add_argument("--huber_m", type=float, default=0.03, help="Huber delta in meters")
    ap.add_argument("--w_pose_reg", type=float, default=1e-3)
    ap.add_argument(
        "--w_temporal",
        type=float,
        default=3e-3,
        help="Temporal pose prior weight when --init_pose_npy is provided.",
    )
    ap.add_argument(
        "--w_temporal_velocity",
        type=float,
        default=0.0,
        help="Weight for temporal velocity target (pose extrapolation from previous 2 frames).",
    )
    ap.add_argument(
        "--w_temporal_accel",
        type=float,
        default=0.0,
        help="Weight for temporal acceleration smoothing when previous 2 frames are available.",
    )
    ap.add_argument("--w_hand_reg", type=float, default=1e-3, help="Regularization weight for optimized hand108 toward initialization.")
    ap.add_argument("--temporal_init_blend", type=float, default=0.7, help="Blend between per-view init and temporal init pose.")
    ap.add_argument("--temporal_extrapolation", type=float, default=1.0, help="Extrapolation gain for temporal velocity target using prev and prev-prev poses.")
    ap.add_argument("--no_optimize_hand_pose", action="store_true", help="Disable hand108 optimization.")
    ap.add_argument("--no_anchor_similarity", action="store_true", help="Disable anchor-only similarity and use all supervised points.")
    ap.add_argument("--topk_print", type=int, default=10)
    ap.add_argument("--no_debug_artifacts", action="store_true", help="Skip debug plots/npz/ply for faster execution.")
    ap.add_argument("--min_iters", type=int, default=50, help="Minimum iterations before early stopping is allowed.")
    ap.add_argument("--early_stop_patience", type=int, default=60, help="Stop after this many non-improving iterations.")
    ap.add_argument("--early_stop_tol", type=float, default=1e-6, help="Minimum loss improvement to reset patience.")
    ap.add_argument("--init_pose_npy", type=Path, default=None, help="Optional .npy with body_pose_params used as temporal initialization.")
    ap.add_argument("--bad_loss_threshold", type=float, default=3e-3, help="Mark result bad when best total loss exceeds this value.")
    ap.add_argument("--bad_data_loss_threshold", type=float, default=2.5e-3, help="Mark result bad when best data loss exceeds this value.")
    ap.add_argument("--bad_loss_growth_ratio", type=float, default=1.5, help="Mark result bad when final_loss / best_loss exceeds this ratio.")
    ap.add_argument("--loss_divergence_ratio", type=float, default=3.0, help="Early break when current loss diverges above best_loss by this ratio.")
    ap.add_argument("--min_valid_points", type=int, default=6, help="Minimum valid points required for Umeyama/loss computations.")
    ap.add_argument(
        "--zero_weight_strategy",
        type=str,
        choices=["uniform_finite", "fail"],
        default="uniform_finite",
        help="Behavior when inlier weights collapse to ~zero on valid points.",
    )
    ap.add_argument(
        "--freeze_lower_body",
        action="store_true",
        help="Freeze lower-body pose dimensions to initialization (reduces stationary-leg jitter).",
    )
    return ap


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> OptimizationConfig:
    return OptimizationConfig(
        npz=args.npz,
        npy_dir=args.npy_dir,
        cams=list(args.cams),
        out_npy=args.out_npy,
        debug_dir=args.debug_dir,
        hf_repo=args.hf_repo,
        ckpt=args.ckpt,
        mhr_pt=args.mhr_pt,
        device=args.device,
        iters=args.iters,
        lr=args.lr,
        with_scale=args.with_scale,
        huber_m=args.huber_m,
        w_pose_reg=args.w_pose_reg,
        w_temporal=args.w_temporal,
        w_temporal_velocity=args.w_temporal_velocity,
        w_temporal_accel=args.w_temporal_accel,
        optimize_hand_pose=not args.no_optimize_hand_pose,
        w_hand_reg=args.w_hand_reg,
        use_anchor_similarity=not args.no_anchor_similarity,
        temporal_init_blend=args.temporal_init_blend,
        temporal_extrapolation=args.temporal_extrapolation,
        topk_print=args.topk_print,
        save_debug_artifacts=not args.no_debug_artifacts,
        min_iters=args.min_iters,
        early_stop_patience=args.early_stop_patience,
        early_stop_tol=args.early_stop_tol,
        init_body_pose=load_body_pose_from_npy(args.init_pose_npy),
        bad_loss_threshold=args.bad_loss_threshold,
        bad_data_loss_threshold=args.bad_data_loss_threshold,
        bad_loss_growth_ratio=args.bad_loss_growth_ratio,
        loss_divergence_ratio=args.loss_divergence_ratio,
        min_valid_points=args.min_valid_points,
        zero_weight_strategy=args.zero_weight_strategy,
        freeze_lower_body=args.freeze_lower_body,
    )


def main(args: Optional[argparse.Namespace] = None) -> OptimizationRunResult:
    if args is None:
        args = parse_args()
    return run_optimization(namespace_to_config(args))


__all__ = [
    "ALIGNMENT_ANCHOR_NAMES",
    "DEFAULT_PARAM_DIMS",
    "MHR_PARAM_HAND_IDXS_133",
    "OptimizationConfig",
    "OptimizationRunResult",
    "OptimizationRuntime",
    "apply_repo_camera_flip_xyz",
    "build_alignment_anchor_local_indices",
    "build_base_keep_mask",
    "build_optimization_runtime",
    "classify_bad_optimization",
    "ensure_dir",
    "find_npy_for_cam",
    "get_param_array",
    "huber",
    "load_body_pose_from_npy",
    "load_sam_head",
    "mhr_fk",
    "plot_3d_compare",
    "plot_loss_curve",
    "resolve_lower_body_pose_indices",
    "resolve_valid_indices_for_prediction",
    "run_optimization",
    "safe_growth",
    "safe_numpy_item_load",
    "sanitize_subset_and_weights",
    "select_alignment_subset_tensors",
    "set_axes_equal",
    "to_torch",
    "umeyama_similarity",
]


if __name__ == "__main__":
    main()
