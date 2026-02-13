# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

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


def resolve_output_folder(image_folder: str, output_folder: str) -> Path:
    if output_folder:
        return Path(output_folder)
    return Path("./output") / Path(image_folder).name


def ensure_output_dirs(output_root: Path, debug: bool) -> tuple[Path, Path, Path]:
    render_root = output_root / "render"
    npy_root = output_root / "npy"
    mesh_root = output_root / "mesh"

    npy_root.mkdir(parents=True, exist_ok=True)
    if debug:
        render_root.mkdir(parents=True, exist_ok=True)
        mesh_root.mkdir(parents=True, exist_ok=True)

    return render_root, npy_root, mesh_root


def resolve_model_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")
    return mhr_path, detector_path, segmentor_path, fov_path


def build_estimator(args: argparse.Namespace) -> SAM3DBodyEstimator:
    mhr_path, detector_path, segmentor_path, fov_path = resolve_model_paths(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path,
        device=device,
        mhr_path=mhr_path,
    )

    human_detector = None
    human_segmentor = None
    fov_estimator = None

    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name,
            device=device,
            path=detector_path,
        )

    should_build_segmentor = (
        args.segmentor_name != "sam2"
        or (args.segmentor_name == "sam2" and len(segmentor_path) > 0)
    )
    if should_build_segmentor:
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name,
            device=device,
            path=segmentor_path,
        )

    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=args.fov_name,
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
    return sorted(
        image
        for ext in IMAGE_EXTENSIONS
        for image in glob(os.path.join(image_root, "**", ext), recursive=True)
    )


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


def save_npy(outputs: Optional[list], output_folder: Path, image_name: str) -> None:
    if not outputs:
        return

    output_dict = outputs[0]
    np.save(output_folder / f"{image_name}.npy", output_dict, allow_pickle=True)


def main(args: argparse.Namespace) -> None:
    output_root = resolve_output_folder(args.image_folder, args.output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    render_root, npy_root, mesh_root = ensure_output_dirs(output_root, args.debug)
    estimator = build_estimator(args)

    images_list = collect_images(args.image_folder)

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        rel_dir = os.path.relpath(os.path.dirname(image_path), args.image_folder)
        if rel_dir == ".":
            rel_dir = ""
        image_name = Path(image_path).stem

        npy_out_dir = npy_root / rel_dir
        npy_out_dir.mkdir(parents=True, exist_ok=True)
        save_npy(outputs, npy_out_dir, image_name)

        if not args.debug:
            continue

        render_out_dir = render_root / rel_dir
        mesh_out_dir = mesh_root / rel_dir
        render_out_dir.mkdir(parents=True, exist_ok=True)
        mesh_out_dir.mkdir(parents=True, exist_ok=True)

        img_cv2 = cv2.imread(image_path)
        ply_files = save_mesh_results(
            img_cv2,
            outputs,
            estimator.faces,
            str(render_out_dir),
            image_name,
        )
        move_mesh_files(ply_files, mesh_out_dir, image_name)


if __name__ == "__main__":
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
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
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

    args = parser.parse_args()
    main(args)
