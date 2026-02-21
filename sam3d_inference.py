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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

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

            img_cv2 = cv2.imread(image_path)
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
    )


def main(args: Optional[argparse.Namespace] = None) -> Demo2RunResult:
    """CLI/programmatic entrypoint for stage-1 inference."""

    if args is None:
        args = parse_args()
    config = namespace_to_config(args)
    return run_demo(config)


if __name__ == "__main__":
    main()
