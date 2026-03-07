#!/usr/bin/env python3
"""Stage-2 triangulation runner with thin orchestration over dedicated helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from studio_hmi_4.common.cv2_compat import require_cv2

from .camera import CaliscopeRig, CameraModel
from .io import (
    NP_EXTS,
    draw_overlay,
    find_existing_with_exts,
    read_image,
    show_3d_scatter_interactive,
)
from .subset import MHRSubset, MHRSubsetSelector
from .triangulation import TriangulatorBA, huber_rho, robust_score


@dataclass
class TriangulationConfig:
    """Configuration for stage-2 triangulation + per-point bundle adjustment."""

    mhr_py: str
    caliscope_toml: str
    cams: List[str]
    npy_dir: str
    out_npz: str
    toml_sections: Optional[List[str]] = None
    index: int = 0
    normalized: bool = False
    pixel: bool = False
    invert_extrinsics: bool = False
    lm_iters: int = 25
    lm_lambda: float = 1e-3
    lm_eps: float = 1e-4
    debug: bool = False
    debug_dir: Optional[str] = None
    img_dir: Optional[str] = None
    score_type: str = "median"
    huber_delta: float = 10.0
    inlier_thresh: float = 30.0
    robust_lm: bool = False
    robust_lm_delta: float = 10.0
    reseed_from_inliers: bool = True


@dataclass
class TriangulationRunResult:
    """Outputs from stage-2 triangulation."""

    out_npz: Path
    points3d_init: np.ndarray
    points3d_refined: np.ndarray
    mean_err_init: Dict[str, float]
    mean_err_refined: Dict[str, float]
    debug_dir: Optional[Path]


def resolve_toml_sections(
    cams: List[str],
    toml_sections_arg: Optional[List[str]],
) -> List[str]:
    if toml_sections_arg is None or len(toml_sections_arg) == 0:
        return cams[:]
    if len(toml_sections_arg) != len(cams):
        raise ValueError("--toml_sections must match length of --cams (or omit).")
    return toml_sections_arg


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhr_py", required=True, help="Path to mhr_70.py (defines pose_info)")
    ap.add_argument("--caliscope_toml", required=True, help="Path to Caliscope config.toml")
    ap.add_argument("--cams", nargs="+", required=True, help="Camera names (= filename stems): front left right")
    ap.add_argument(
        "--toml_sections",
        nargs="*",
        default=None,
        help="Optional TOML section names aligned with --cams: e.g. cam_1 cam_2 cam_3",
    )
    ap.add_argument("--npy_dir", required=True, help="Directory with per-cam npy/npz (front.npy, ...)")
    ap.add_argument("--out_npz", required=True, help="Output .npz path")
    ap.add_argument("--index", type=int, default=0, help="Frame index if pred_keypoints_2d is (N,70,2)")
    ap.add_argument("--normalized", action="store_true", help="Force treat input 2D as normalized [0,1]")
    ap.add_argument("--pixel", action="store_true", help="Force treat input 2D as pixel coords")
    ap.add_argument(
        "--invert_extrinsics",
        action="store_true",
        help="Try if reprojection is wrong (treat stored extrinsics as cam->world)",
    )
    ap.add_argument("--lm_iters", type=int, default=25, help="LM iterations per point")
    ap.add_argument("--lm_lambda", type=float, default=1e-3, help="Initial LM damping")
    ap.add_argument("--lm_eps", type=float, default=1e-4, help="Finite-difference step scale")
    ap.add_argument("--debug", action="store_true", help="Show interactive 3D scatter (refined points)")
    ap.add_argument(
        "--debug_dir",
        default=None,
        help="If set, save overlay images and a 3D scatter png (no interactive 2D)",
    )
    ap.add_argument("--img_dir", default=None, help="Optional: image dir for saving overlays (front.jpg, ...)")
    ap.add_argument(
        "--score_type",
        type=str,
        default="median",
        choices=["median", "trimmed", "huber"],
        help="Robust candidate scoring for pair init: median | trimmed | huber",
    )
    ap.add_argument("--huber_delta", type=float, default=10.0, help="Delta for huber score (pixels)")
    ap.add_argument("--inlier_thresh", type=float, default=30.0, help="Inlier threshold tau (pixels) for selecting views before BA")
    ap.add_argument("--robust_lm", action="store_true", help="Use Huber-weighted residuals inside LM")
    ap.add_argument("--robust_lm_delta", type=float, default=10.0, help="Delta for robust LM Huber weighting (pixels)")
    ap.add_argument(
        "--no_reseed_from_inliers",
        action="store_true",
        help="Disable DLT reseeding from selected inlier views before LM refinement.",
    )
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> TriangulationConfig:
    return TriangulationConfig(
        mhr_py=args.mhr_py,
        caliscope_toml=args.caliscope_toml,
        cams=list(args.cams),
        toml_sections=args.toml_sections,
        npy_dir=args.npy_dir,
        out_npz=args.out_npz,
        index=args.index,
        normalized=args.normalized,
        pixel=args.pixel,
        invert_extrinsics=args.invert_extrinsics,
        lm_iters=args.lm_iters,
        lm_lambda=args.lm_lambda,
        lm_eps=args.lm_eps,
        debug=args.debug,
        debug_dir=args.debug_dir,
        img_dir=args.img_dir,
        score_type=args.score_type,
        huber_delta=args.huber_delta,
        inlier_thresh=args.inlier_thresh,
        robust_lm=args.robust_lm,
        robust_lm_delta=args.robust_lm_delta,
        reseed_from_inliers=not args.no_reseed_from_inliers,
    )


def run_triangulation(config: TriangulationConfig) -> TriangulationRunResult:
    cams = list(config.cams)
    if len(cams) < 2:
        raise ValueError("Need at least 2 cameras for triangulation/BA.")

    toml_sections = resolve_toml_sections(cams, config.toml_sections)
    out_npz = Path(config.out_npz).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if config.debug_dir is not None:
        debug_dir = Path(config.debug_dir).expanduser().resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(config.img_dir).expanduser().resolve() if config.img_dir is not None else None

    subset = MHRSubsetSelector(config.mhr_py).build_subset()
    cameras = CaliscopeRig(config.caliscope_toml).build_cameras(
        cams=cams,
        toml_sections=toml_sections,
        invert_extrinsics=config.invert_extrinsics,
    )

    pipe = TriangulatorBA(
        cams=cams,
        cameras=cameras,
        subset=subset,
        npy_dir=config.npy_dir,
        index=config.index,
        force_normalized=config.normalized,
        force_pixel=config.pixel,
        lm_iters=config.lm_iters,
        lm_lambda=config.lm_lambda,
        lm_eps=config.lm_eps,
        score_type=config.score_type,
        huber_delta=config.huber_delta,
        inlier_thresh=config.inlier_thresh,
        robust_lm=config.robust_lm,
        robust_lm_delta=config.robust_lm_delta,
        reseed_from_inliers=config.reseed_from_inliers,
    )
    pipe.load_observations()

    points3d_init, best_pair_idx, inlier_mask = pipe.init_by_pair_selection()
    points3d_ref = pipe.refine_lm_per_point(points3d_init, inlier_mask)
    proj_init, mean_init = pipe.per_cam_reprojection(points3d_init)
    proj_ref, mean_ref = pipe.per_cam_reprojection(points3d_ref)
    err_init = pipe.per_cam_errors(points3d_init, proj_init)
    err_ref = pipe.per_cam_errors(points3d_ref, proj_ref)

    pipe.save_npz(
        out_npz=out_npz,
        points3d_init=points3d_init,
        points3d_ref=points3d_ref,
        best_pair_idx=best_pair_idx,
        inlier_mask=inlier_mask,
        mean_init=mean_init,
        mean_ref=mean_ref,
        proj_init=proj_init,
        proj_ref=proj_ref,
        err_init=err_init,
        err_ref=err_ref,
        toml_sections=toml_sections,
    )

    print(f"[OK] Saved: {out_npz}")
    print("Mean reprojection error (px):")
    for cam in cams:
        print(f"  {cam}: init={mean_init[cam]:.2f}  refined={mean_ref[cam]:.2f}")

    if debug_dir is not None:
        cv2 = require_cv2("stage-2 debug overlay writing")
        for cam in cams:
            camera = cameras[cam]
            w_img, h_img = camera.w, camera.h
            if img_dir is not None:
                img = read_image(img_dir, cam, fallback_size_wh=(w_img, h_img))
            else:
                img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

            obs = pipe.obs_per_cam[cam]
            over_init = draw_overlay(img, obs, proj_init[cam], subset.edges)
            cv2.putText(
                over_init,
                f"{cam} init mean_err={mean_init[cam]:.2f}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(debug_dir / f"{cam}_overlay_init.jpg"), over_init)

            over_ref = draw_overlay(img, obs, proj_ref[cam], subset.edges)
            cv2.putText(
                over_ref,
                f"{cam} refined mean_err={mean_ref[cam]:.2f}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(debug_dir / f"{cam}_overlay_refined.jpg"), over_ref)

        try:
            import matplotlib.pyplot as plt

            X = points3d_ref
            ok = np.isfinite(X).all(axis=1)
            X = X[ok]
            if X.shape[0] > 0:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=12)
                ax.set_title("Triangulated 3D (refined)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                fig.tight_layout()
                fig.savefig(str(debug_dir / "triangulated_3d_refined.png"), dpi=160)
                plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Could not save 3D scatter PNG: {exc}")

        print(f"[DEBUG] Saved debug files in: {debug_dir}")

    if config.debug:
        show_3d_scatter_interactive(points3d_ref, title="Triangulated 3D (refined)")

    return TriangulationRunResult(
        out_npz=out_npz,
        points3d_init=points3d_init,
        points3d_refined=points3d_ref,
        mean_err_init=mean_init,
        mean_err_refined=mean_ref,
        debug_dir=debug_dir,
    )


def main(args: Optional[argparse.Namespace] = None) -> TriangulationRunResult:
    if args is None:
        args = parse_args()
    return run_triangulation(namespace_to_config(args))


__all__ = [
    "CameraModel",
    "CaliscopeRig",
    "MHRSubset",
    "MHRSubsetSelector",
    "NP_EXTS",
    "TriangulationConfig",
    "TriangulationRunResult",
    "TriangulatorBA",
    "find_existing_with_exts",
    "huber_rho",
    "robust_score",
    "run_triangulation",
]


if __name__ == "__main__":
    main()
