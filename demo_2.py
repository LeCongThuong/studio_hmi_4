# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob
import shutil 

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
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import (
    visualize_sample,
    visualize_sample_together,
)
from tools.utils import save_mesh_results 
from tqdm import tqdm


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # NEW: enforce dest_dir structure
    render_root = os.path.join(output_folder, "render")
    npy_root = os.path.join(output_folder, "npy")
    mesh_root = os.path.join(output_folder, "mesh")  # NEW
    os.makedirs(npy_root, exist_ok=True)
    if args.debug:
        os.makedirs(render_root, exist_ok=True)
        os.makedirs(mesh_root, exist_ok=True)  # NEW

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )

    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]

    # NEW: recursive glob under root_dir (args.image_folder)
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, "**", ext), recursive=True)
        ]
    )

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )
        # print("Key of outputs[0]:", outputs[0].keys())
        # NEW: keep subfolder structure in output (dest_dir/{render,npy,mesh}/<rel_dir>/)
        rel_dir = os.path.relpath(os.path.dirname(image_path), args.image_folder)
        if rel_dir == ".":
            rel_dir = ""
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        npy_out_dir = os.path.join(npy_root, rel_dir)
        os.makedirs(npy_out_dir, exist_ok=True)
        save_npy(outputs, npy_out_dir, image_name)

        # NEW: render + mesh only if --debug
        if args.debug:
            render_out_dir = os.path.join(render_root, rel_dir)
            mesh_out_dir = os.path.join(mesh_root, rel_dir)  # NEW
            os.makedirs(render_out_dir, exist_ok=True)
            os.makedirs(mesh_out_dir, exist_ok=True)  # NEW

            img_cv2 = cv2.imread(image_path)

            # Save all results (PLY meshes, overlay images, bbox images)
            # NOTE: we write extras into render_out_dir, then move PLY(s) to mesh_out_dir
            ply_files = save_mesh_results(
                img_cv2, outputs, estimator.faces, render_out_dir, image_name
            )

            # Move PLY(s) into mesh/<rel_dir>/ (and rename if only 1 person)
            if ply_files is not None:
                if isinstance(ply_files, (list, tuple)):
                    ply_list = list(ply_files)
                else:
                    ply_list = [ply_files]

                # keep only existing .ply files
                ply_paths = []
                for p in ply_list:
                    if p is None:
                        continue
                    src = str(p)
                    if os.path.isfile(src) and src.lower().endswith(".ply"):
                        ply_paths.append(src)

                if len(ply_paths) == 1:
                    # single person => single mesh => rename to <image_name>.ply
                    src = ply_paths[0]
                    dst = os.path.join(mesh_out_dir, f"{image_name}.ply")
                    shutil.move(src, dst)
                else:
                    # multiple => keep original names
                    for src in ply_paths:
                        shutil.move(src, os.path.join(mesh_out_dir, os.path.basename(src)))

def save_npy(outputs, output_folder, image_name):
    if outputs is None or len(outputs) == 0:
        return

    output_dict = outputs[0]
    
    # Save as a single .npy (stores the whole dict as an object)
    np.save(os.path.join(output_folder, f"{image_name}.npy"), output_dict, allow_pickle=True)


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

    # NEW: debug flag for rendering/mesh export
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="If set, save render + mesh into <output_folder>/{render,mesh}/. If not set, only .npy is saved.",
    )

    args = parser.parse_args()

    main(args)
