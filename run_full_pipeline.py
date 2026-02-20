#!/usr/bin/env python3
"""Unified end-to-end runner for SAM-3D body fitting across all pipeline stages.

Pipeline stages:
1. Stage-1 (`sam3d_inference.py`): infer per-view 2D/3D outputs from images.
2. Stage-2 (`triangulate_mhr3d_gt.py`): triangulate subset 3D GT with robust BA.
3. Stage-3 (`optimize_mhr_pose.py`): optimize MHR pose parameters to triangulated GT.

The runner supports both one-shot execution and partial reruns via `--skip_*` flags.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from sam3d_inference import Demo2Config, run_demo
from optimize_mhr_pose import (
    OptimizationConfig,
    build_optimization_runtime,
    run_optimization,
)
from triangulate_mhr3d_gt import (
    NP_EXTS,
    TriangulationConfig,
    find_existing_with_exts,
    run_triangulation,
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
    topk_print: int = 10
    save_opt_debug: bool = False


@dataclass
class FramePipelineResult:
    """Per-frame summary produced by the full pipeline."""

    rel_dir: str
    npy_dir: Path
    triangulated_npz: Path
    optimized_npy: Optional[Path]


@dataclass
class FullPipelineResult:
    """Aggregated output of the full pipeline run."""

    output_root: Path
    npy_root: Path
    frames: List[FramePipelineResult]


def _dir_has_all_cam_predictions(dir_path: Path, cams: Sequence[str]) -> bool:
    for cam in cams:
        if find_existing_with_exts(dir_path, cam, NP_EXTS) is None:
            return False
    return True


def discover_frame_dirs(
    npy_root: Path,
    cams: Sequence[str],
    frame_rel: Optional[str] = None,
) -> List[Path]:
    """Discover frame directories containing one prediction file per camera."""

    if frame_rel:
        target = (npy_root / frame_rel).resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"Requested frame_rel directory not found: {target}")
        if not _dir_has_all_cam_predictions(target, cams):
            raise FileNotFoundError(
                f"Directory does not contain all camera predictions {list(cams)}: {target}"
            )
        return [target]

    candidates = [npy_root]
    candidates.extend(sorted(p for p in npy_root.rglob("*") if p.is_dir()))
    return [p for p in candidates if _dir_has_all_cam_predictions(p, cams)]


def _rel_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if str(rel) == ".":
        return ""
    return rel.as_posix()


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

    # Pipeline control
    ap.add_argument("--frame_rel", default=None, type=str, help="Optional relative frame dir under npy root.")
    ap.add_argument("--skip_inference", action="store_true", default=False)
    ap.add_argument("--skip_triangulation", action="store_true", default=False)
    ap.add_argument("--skip_optimization", action="store_true", default=False)
    ap.add_argument("--overwrite", action="store_true", default=False, help="Recompute outputs even when they already exist.")
    ap.add_argument("--npy_root", default=None, type=str, help="Existing npy root (used when --skip_inference).")
    ap.add_argument("--triangulated_name", default="triangulated.npz", type=str)
    ap.add_argument("--optimized_name", default="opt_out.npy", type=str)

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
    ap.add_argument("--topk_print", type=int, default=10)
    ap.add_argument("--save_opt_debug", action="store_true", default=False, help="Save optimization debug plots/artifacts.")
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args for full pipeline execution."""

    parser = build_arg_parser()
    return parser.parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> FullPipelineConfig:
    """Convert parsed CLI args to `FullPipelineConfig`."""

    return FullPipelineConfig(
        image_folder=args.image_folder,
        output_root=args.output_root,
        cams=list(args.cams),
        caliscope_toml=args.caliscope_toml,
        mhr_py=args.mhr_py,
        toml_sections=args.toml_sections,
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
        detector_name=args.detector_name,
        segmentor_name=args.segmentor_name,
        fov_name=args.fov_name,
        detector_path=args.detector_path,
        segmentor_path=args.segmentor_path,
        fov_path=args.fov_path,
        bbox_thresh=args.bbox_thresh,
        use_mask=args.use_mask,
        debug_inference=args.debug_inference,
        save_mhr_params=args.save_mhr_params,
        frame_rel=args.frame_rel,
        skip_inference=args.skip_inference,
        skip_triangulation=args.skip_triangulation,
        skip_optimization=args.skip_optimization,
        overwrite=args.overwrite,
        npy_root=args.npy_root,
        triangulated_name=args.triangulated_name,
        optimized_name=args.optimized_name,
        normalized=args.normalized,
        pixel=args.pixel,
        invert_extrinsics=args.invert_extrinsics,
        lm_iters=args.lm_iters,
        lm_lambda=args.lm_lambda,
        lm_eps=args.lm_eps,
        score_type=args.score_type,
        huber_delta=args.huber_delta,
        inlier_thresh=args.inlier_thresh,
        robust_lm=args.robust_lm,
        robust_lm_delta=args.robust_lm_delta,
        debug_triangulation=args.debug_triangulation,
        save_triangulation_debug=args.save_triangulation_debug,
        hf_repo=args.hf_repo,
        opt_ckpt=args.opt_ckpt,
        opt_mhr_pt=args.opt_mhr_pt,
        device=args.device,
        iters=args.iters,
        lr=args.lr,
        with_scale=args.with_scale,
        huber_m=args.huber_m,
        w_pose_reg=args.w_pose_reg,
        topk_print=args.topk_print,
        save_opt_debug=args.save_opt_debug,
    )


def run_full_pipeline(config: FullPipelineConfig) -> FullPipelineResult:
    """Run all enabled stages and return per-frame outputs.

    Core idea:
    - Stage-1 creates/loads per-camera SAM predictions.
    - Frame folders are discovered under the stage-1 `npy` tree.
    - Stage-2 and stage-3 run per frame folder and write deterministic outputs.
    """

    image_root = Path(config.image_folder).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    inference_root = output_root / "inference"
    triangulation_root = output_root / "triangulation"
    optimization_root = output_root / "optimization"
    output_root.mkdir(parents=True, exist_ok=True)

    if not config.skip_inference and not config.checkpoint_path:
        raise ValueError("--checkpoint_path is required unless --skip_inference is set.")

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
        reuse_stage1 = False
        if (
            not config.overwrite
            and config.frame_rel is not None
            and inferred_npy_root.is_dir()
        ):
            rel_dir = (inferred_npy_root / config.frame_rel).resolve()
            if rel_dir.is_dir() and _dir_has_all_cam_predictions(rel_dir, config.cams):
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
            )
            demo_result = run_demo(demo_cfg)
            npy_root = demo_result.npy_root.resolve()

    frame_dirs = discover_frame_dirs(
        npy_root=npy_root,
        cams=config.cams,
        frame_rel=config.frame_rel,
    )
    if not frame_dirs:
        raise FileNotFoundError(
            f"No frame directories with all cams {config.cams} found under {npy_root}"
        )

    opt_runtime = None
    if not config.skip_optimization:
        runtime_cfg = OptimizationConfig(
            npz=Path("runtime.npz"),
            npy_dir=frame_dirs[0],
            cams=list(config.cams),
            out_npy=Path("runtime.npy"),
            debug_dir=(optimization_root / "debug_opt"),
            hf_repo=config.hf_repo,
            ckpt=opt_ckpt,
            mhr_pt=opt_mhr_pt,
            device=config.device,
            save_debug_artifacts=False,
        )
        opt_runtime = build_optimization_runtime(runtime_cfg)

    frame_results: List[FramePipelineResult] = []
    for idx, frame_npy_dir in enumerate(frame_dirs, start=1):
        rel_dir = _rel_key(frame_npy_dir, npy_root)
        print(f"[PIPELINE] Frame {idx}/{len(frame_dirs)} rel='{rel_dir or '.'}'")

        tri_out = (triangulation_root / rel_dir / config.triangulated_name).resolve()
        tri_out.parent.mkdir(parents=True, exist_ok=True)
        tri_debug_dir = (tri_out.parent / "debug") if config.save_triangulation_debug else None

        rel_img_dir = image_root / rel_dir
        img_dir = rel_img_dir if rel_img_dir.is_dir() else image_root

        if not config.skip_triangulation:
            if tri_out.exists() and not config.overwrite:
                print(f"[PIPELINE] Reusing triangulation: {tri_out}")
            else:
                tri_cfg = TriangulationConfig(
                    mhr_py=config.mhr_py,
                    caliscope_toml=config.caliscope_toml,
                    cams=list(config.cams),
                    toml_sections=config.toml_sections,
                    npy_dir=str(frame_npy_dir),
                    out_npz=str(tri_out),
                    normalized=config.normalized,
                    pixel=config.pixel,
                    invert_extrinsics=config.invert_extrinsics,
                    lm_iters=config.lm_iters,
                    lm_lambda=config.lm_lambda,
                    lm_eps=config.lm_eps,
                    debug=config.debug_triangulation,
                    debug_dir=str(tri_debug_dir) if tri_debug_dir else None,
                    img_dir=str(img_dir),
                    score_type=config.score_type,
                    huber_delta=config.huber_delta,
                    inlier_thresh=config.inlier_thresh,
                    robust_lm=config.robust_lm,
                    robust_lm_delta=config.robust_lm_delta,
                )
                run_triangulation(tri_cfg)
        elif not tri_out.exists():
            raise FileNotFoundError(f"Missing triangulation output while --skip_triangulation: {tri_out}")

        optimized_npy: Optional[Path] = None
        if not config.skip_optimization:
            optimized_npy = (optimization_root / rel_dir / config.optimized_name).resolve()
            if optimized_npy.exists() and not config.overwrite:
                print(f"[PIPELINE] Reusing optimized output: {optimized_npy}")
            else:
                opt_cfg = OptimizationConfig(
                    npz=tri_out,
                    npy_dir=frame_npy_dir,
                    cams=list(config.cams),
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
                    topk_print=config.topk_print,
                    save_debug_artifacts=config.save_opt_debug,
                )
                run_optimization(opt_cfg, runtime=opt_runtime)

        frame_results.append(
            FramePipelineResult(
                rel_dir=rel_dir,
                npy_dir=frame_npy_dir,
                triangulated_npz=tri_out,
                optimized_npy=optimized_npy,
            )
        )

    return FullPipelineResult(
        output_root=output_root,
        npy_root=npy_root,
        frames=frame_results,
    )


def main(args: Optional[argparse.Namespace] = None) -> FullPipelineResult:
    """CLI/programmatic entrypoint for full pipeline execution."""

    if args is None:
        args = parse_args()
    config = namespace_to_config(args)
    return run_full_pipeline(config)


if __name__ == "__main__":
    main()
