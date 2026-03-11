#!/usr/bin/env python3
"""Thin CLI/API surface for end-to-end sequence orchestration."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from studio_hmi_4.sequence.presets import public_full_pipeline_defaults
from studio_hmi_4.stage1.runner import Demo2Config, run_demo
from studio_hmi_4.stage2.runner import NP_EXTS, TriangulationConfig, find_existing_with_exts, run_triangulation
from studio_hmi_4.stage3.runner import (
    OptimizationConfig,
    OptimizationRunResult,
    build_optimization_runtime,
    run_optimization,
)

from .orchestrator import run_full_pipeline as _run_full_pipeline_impl
from .types import FrameInputEntry, FramePipelineResult, FullPipelineConfig, FullPipelineResult


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run SAM -> triangulation -> optimization as one full pipeline.")
    ap.add_argument("--image_folder", required=True, type=str, help="Input image root.")
    ap.add_argument("--output_root", required=True, type=str, help="Pipeline output root.")
    ap.add_argument("--cams", nargs="+", required=True, help="Camera names (file stems), e.g. left front right.")
    ap.add_argument("--caliscope_toml", required=True, type=str, help="Path to Caliscope TOML.")
    ap.add_argument("--checkpoint_path", required=True, type=str, help="SAM-3D checkpoint for stage-1.")
    ap.add_argument("--mhr_path", required=True, type=str, help="MHR model path for stage-1.")

    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--hf_repo", type=str, default=None)
    grp.add_argument("--opt_ckpt", type=str, default=None)
    ap.add_argument("--opt_mhr_pt", type=str, default="", help="MHR model path when using --opt_ckpt.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--frame_rel", default=None, type=str, help="Optional relative frame dir under inferred npy root.")
    ap.add_argument("--overwrite", action="store_true", default=False, help="Recompute outputs even when they already exist.")
    ap.add_argument(
        "--skip_inference",
        action="store_true",
        default=False,
        help="Skip stage-1 inference and reuse existing outputs from <output_root>/inference/npy.",
    )
    ap.add_argument(
        "--skip_triangulation",
        action="store_true",
        default=False,
        help="Skip stage-2 triangulation and reuse existing outputs from <output_root>/triangulation.",
    )
    ap.add_argument("--min_views", type=int, default=2, help="Minimum available views required per frame.")
    ap.add_argument(
        "--enable_specialized_hand_fusion",
        action="store_true",
        default=False,
        help="Enable specialized hand-model fusion into stage-1 pred_keypoints_2d.",
    )
    ap.add_argument(
        "--specialized_hand_input_root",
        type=str,
        default="",
        help="Root directory of precomputed hand detections. Expected files: <root>/<rel_dir>/<image_name>.npy|.npz|.json",
    )
    ap.add_argument(
        "--fixed_mhr_param_frame_idx",
        type=int,
        default=None,
        help="Optional frame index used as fixed source for non-pose MHR params (scale/shape/expr).",
    )
    ap.add_argument(
        "--fixed_mhr_param_cam",
        type=str,
        default="front",
        help="Camera name used to read fixed non-pose MHR params from --fixed_mhr_param_frame_idx.",
    )
    ap.add_argument("--freeze_lower_body", action="store_true", help="Freeze lower-body pose dimensions in stage-3 optimization.")
    ap.add_argument("--save_sequence_mp4", action="store_true", help="Export sequence debug MP4.")
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> FullPipelineConfig:
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
        raise AttributeError("Missing required args for namespace_to_config: " + ", ".join(missing_required))
    if not isinstance(cams, (list, tuple)):
        raise AttributeError("Argument --cams must be a sequence of camera names.")

    defaults = public_full_pipeline_defaults()
    enable_hand_fusion = bool(getattr(args, "enable_specialized_hand_fusion", False))
    config_kwargs = dict(defaults)
    config_kwargs.update(
        {
            "image_folder": str(image_folder),
            "output_root": str(output_root),
            "cams": list(cams),
            "caliscope_toml": str(caliscope_toml),
            "checkpoint_path": str(getattr(args, "checkpoint_path", "")),
            "mhr_path": str(getattr(args, "mhr_path", "")),
            "enable_specialized_hand_fusion": enable_hand_fusion,
            "specialized_hand_model": "wilor" if enable_hand_fusion else "none",
            "specialized_hand_input_root": str(getattr(args, "specialized_hand_input_root", "")),
            "frame_rel": getattr(args, "frame_rel", None),
            "overwrite": bool(getattr(args, "overwrite", False)),
            "skip_inference": bool(getattr(args, "skip_inference", False)),
            "skip_triangulation": bool(getattr(args, "skip_triangulation", False)),
            "hf_repo": getattr(args, "hf_repo", None),
            "opt_ckpt": getattr(args, "opt_ckpt", None),
            "opt_mhr_pt": str(getattr(args, "opt_mhr_pt", "")),
            "device": getattr(args, "device", "cuda"),
            "fixed_mhr_param_frame_idx": getattr(args, "fixed_mhr_param_frame_idx", None),
            "fixed_mhr_param_cam": str(getattr(args, "fixed_mhr_param_cam", "front")),
            "freeze_lower_body": bool(getattr(args, "freeze_lower_body", False)),
            "min_views": int(getattr(args, "min_views", 2)),
            "save_sequence_mp4": bool(getattr(args, "save_sequence_mp4", False)),
        }
    )
    return FullPipelineConfig(**config_kwargs)


def run_full_pipeline(config: FullPipelineConfig) -> FullPipelineResult:
    return _run_full_pipeline_impl(
        config,
        run_demo_fn=run_demo,
        run_triangulation_fn=run_triangulation,
        build_optimization_runtime_fn=build_optimization_runtime,
        run_optimization_fn=run_optimization,
    )


def main(args: Optional[argparse.Namespace] = None) -> FullPipelineResult:
    if args is None:
        args = parse_args()
    return run_full_pipeline(namespace_to_config(args))


__all__ = [
    "Demo2Config",
    "FrameInputEntry",
    "FramePipelineResult",
    "FullPipelineConfig",
    "FullPipelineResult",
    "NP_EXTS",
    "OptimizationConfig",
    "OptimizationRunResult",
    "TriangulationConfig",
    "build_arg_parser",
    "build_optimization_runtime",
    "find_existing_with_exts",
    "main",
    "namespace_to_config",
    "parse_args",
    "run_demo",
    "run_full_pipeline",
    "run_optimization",
    "run_triangulation",
]


if __name__ == "__main__":
    main()
