# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Stage-1 runner for SAM-3D inference and per-frame artifact persistence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

try:
    import pyrootutils
except Exception:  # pragma: no cover
    pyrootutils = None

if pyrootutils is not None:  # pragma: no branch
    pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml", ".sl"],
        pythonpath=True,
        dotenv=True,
    )

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):  # type: ignore
        return it

from studio_hmi_4.common import relative_dir as _relative_dir
from studio_hmi_4.common import validate_stage1_prediction_dict
from studio_hmi_4.common.cv2_compat import require_cv2
from studio_hmi_4.stage1.fusion import (
    apply_specialized_hand_fusion_live,
    apply_specialized_hand_fusion_precomputed,
    build_specialized_hand_estimator,
    extract_k70_keypoints,
    render_specialized_hand_debug_vis,
)
from studio_hmi_4.stage1.io import (
    MHR_PARAM_KEYS,
    collect_images,
    ensure_output_dirs,
    extract_mhr_params,
    extract_primary_output,
    filter_images_by_rel_dirs,
    move_mesh_files,
    resolve_model_paths,
    resolve_output_folder,
    save_dict_npy,
)

try:
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.utils import save_mesh_results
except Exception:  # pragma: no cover
    SAM3DBodyEstimator = Any  # type: ignore[assignment]
    load_sam_3d_body = None
    save_mesh_results = None


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


def build_estimator(config: Demo2Config) -> SAM3DBodyEstimator:
    """Construct the SAM-3D estimator and optional auxiliary models."""

    if load_sam_3d_body is None:
        raise ImportError(
            "sam_3d_body is not importable in this environment. "
            "Pass a fake estimator for tests or install the SAM-3D dependencies."
        )

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


def _stage1_meta(config: Demo2Config, precomputed_hand_root: Optional[Path]) -> dict[str, object]:
    return {
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
            None if precomputed_hand_root is None else str(precomputed_hand_root)
        ),
        "specialized_hand_detector_conf": float(config.specialized_hand_detector_conf),
        "specialized_hand_rescale_factor": float(config.specialized_hand_rescale_factor),
        "specialized_hand_wrist_max_dist_px": float(config.specialized_hand_wrist_max_dist_px),
        "replace_wrist_with_specialized": bool(config.replace_wrist_with_specialized),
        "specialized_hand_debug_vis": bool(config.specialized_hand_debug_vis),
        "specialized_hand_debug_dirname": str(config.specialized_hand_debug_dirname),
        "wilor_repo_id": str(config.wilor_repo_id),
    }


def run_demo(
    config: Demo2Config,
    estimator: Optional[SAM3DBodyEstimator] = None,
    show_progress: bool = True,
) -> Demo2RunResult:
    """Execute stage-1 inference for all images under `config.image_folder`."""

    output_root = resolve_output_folder(config.image_folder, config.output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    render_root, npy_root, mesh_root, mhr_root = ensure_output_dirs(
        output_root=output_root,
        debug=config.debug,
        save_mhr_params=config.save_mhr_params,
    )
    if estimator is None:
        estimator = build_estimator(config)

    hand_estimator = build_specialized_hand_estimator(config)
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
        images_list = filter_images_by_rel_dirs(
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

        pre_fusion_k70 = None
        if output_dict is not None and bool(config.enable_specialized_hand_fusion):
            pre_fusion_k70, _ = extract_k70_keypoints(output_dict)

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
            img_cv2 = require_cv2("stage-1 image loading").imread(image_path)

        if output_dict is not None and bool(config.enable_specialized_hand_fusion):
            if hand_source in {"precomputed", "file", "files"}:
                assert precomputed_hand_root is not None
                output_dict = apply_specialized_hand_fusion_precomputed(
                    output_dict=output_dict,
                    hand_root=precomputed_hand_root,
                    rel_dir=rel_dir,
                    image_name=image_name,
                    config=config,
                )
            elif hand_source in {"live", "runtime"} and hand_estimator is not None and img_cv2 is not None:
                output_dict = apply_specialized_hand_fusion_live(
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
            post_fusion_k70, _ = extract_k70_keypoints(output_dict)
            if post_fusion_k70 is not None:
                source_mask = output_dict.get("pred_keypoints_2d_source", torch.zeros(70, dtype=torch.int8).numpy())
                vis = render_specialized_hand_debug_vis(
                    image_bgr=img_cv2,
                    pre_k70=pre_fusion_k70,
                    post_k70=post_fusion_k70,
                    source_mask=source_mask,
                    output_dict=output_dict,
                )
                vis_out_dir = hand_debug_root / rel_dir
                vis_out_dir.mkdir(parents=True, exist_ok=True)
                require_cv2("stage-1 specialized-hand debug image writing").imwrite(
                    str(vis_out_dir / f"{image_name}.jpg"),
                    vis,
                )

        npy_out_dir = npy_root / rel_dir
        npy_out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = None
        if output_dict is not None:
            validate_stage1_prediction_dict(output_dict, require_pose_blocks=False)
            npy_path = save_dict_npy(output_dict, npy_out_dir, image_name)

        mhr_params_path = None
        if mhr_root is not None and output_dict is not None:
            mhr_out_dir = mhr_root / rel_dir
            mhr_out_dir.mkdir(parents=True, exist_ok=True)
            mhr_params = extract_mhr_params(output_dict, strict=False, keys=MHR_PARAM_KEYS)
            if mhr_params:
                mhr_params_path = save_dict_npy(mhr_params, mhr_out_dir, image_name)

        if config.debug and output_dict is not None:
            if save_mesh_results is None:
                raise ImportError(
                    "tools.utils.save_mesh_results is unavailable; Stage-1 debug mesh export "
                    "requires the SAM-3D runtime environment."
                )
            render_out_dir = render_root / rel_dir
            mesh_out_dir = mesh_root / rel_dir
            render_out_dir.mkdir(parents=True, exist_ok=True)
            mesh_out_dir.mkdir(parents=True, exist_ok=True)

            if img_cv2 is not None:
                debug_outputs = outputs if isinstance(outputs, (list, tuple)) else [output_dict]
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

    (output_root / "stage1_meta.json").write_text(
        json.dumps(_stage1_meta(config, precomputed_hand_root), indent=2),
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
    parser.add_argument("--image_folder", required=True, type=str, help="Path to folder containing input images")
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to SAM 3D Body model checkpoint")
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
    parser.add_argument("--fov_path", default="", type=str, help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)")
    parser.add_argument("--mhr_path", default="", type=str, help="Path to MoHR/assets folder (or set SAM3D_MHR_PATH)")
    parser.add_argument("--bbox_thresh", default=0.8, type=float, help="Bounding box detection threshold")
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
    parser.add_argument("--person_index", type=int, default=0, help="Person index to use when --person_select_strategy=person_index.")
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
        help="Root directory of precomputed hand detections. Expected files: <root>/<rel_dir>/<image_name>.npy|.npz|.json",
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
    parser.add_argument("--wilor_pretrained_dir", type=str, default="", help="Optional local directory for WiLoR-mini pretrained assets.")
    parser.add_argument(
        "--wilor_repo_id",
        type=str,
        default="warmshao/WiLoR-mini",
        help="Hugging Face repo id used by WiLoR-mini for pretrained assets.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> Demo2Config:
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
    if args is None:
        args = parse_args()
    return run_demo(namespace_to_config(args))


__all__ = [
    "Demo2Config",
    "Demo2RunResult",
    "FrameResult",
    "MHR_PARAM_KEYS",
    "SAM3DBodyEstimator",
    "build_estimator",
    "collect_images",
    "extract_mhr_params",
    "extract_primary_output",
    "filter_images_by_rel_dirs",
    "run_demo",
]


if __name__ == "__main__":
    main()
