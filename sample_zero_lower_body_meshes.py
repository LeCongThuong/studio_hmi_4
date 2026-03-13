#!/usr/bin/env python3
"""Generate random MHR meshes with lower-body pose dimensions fixed to zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from studio_hmi_4.common import save_debug_mesh_png, save_npy_dict, write_ply
from studio_hmi_4.stage3.alignment import resolve_lower_body_pose_indices
from studio_hmi_4.stage3.runtime import (
    apply_repo_camera_flip_xyz,
    load_sam_head,
    mhr_fk,
    to_torch,
)
from studio_hmi_4.stage3.types import OptimizationConfig


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Sample random MHR meshes while forcing the repo's hardcoded lower-body "
            "pose indices to zero."
        )
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hf_repo", type=str, default=None, help="HF repo used to load the SAM/MHR head.")
    grp.add_argument("--ckpt", type=str, default=None, help="Checkpoint path used to load the SAM/MHR head.")
    ap.add_argument("--mhr_pt", type=str, default="", help="MHR model path (required with --ckpt).")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_dir", required=True, type=Path, help="Directory that will receive meshes and params.")
    ap.add_argument("--k", type=int, default=8, help="Number of random samples to generate.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for numpy RNG.")
    ap.add_argument(
        "--pose_dim",
        type=int,
        default=133,
        choices=[133, 204],
        help="Body-pose parameter dimensionality to sample.",
    )
    ap.add_argument("--body_std", type=float, default=0.35, help="Std-dev for random body-pose coefficients.")
    ap.add_argument("--hand_std", type=float, default=0.35, help="Std-dev for random hand-pose coefficients.")
    ap.add_argument("--scale_std", type=float, default=0.10, help="Std-dev for random scale coefficients.")
    ap.add_argument("--shape_std", type=float, default=0.35, help="Std-dev for random shape coefficients.")
    ap.add_argument("--expr_std", type=float, default=0.20, help="Std-dev for random expression coefficients.")
    ap.add_argument(
        "--clip",
        type=float,
        default=1.5,
        help="Optional symmetric clip applied to all random coefficients (<=0 disables clipping).",
    )
    ap.add_argument("--save_png", action="store_true", default=False, help="Also save simple mesh preview PNGs.")
    return ap


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def build_head(args: argparse.Namespace):
    cfg = OptimizationConfig(
        npz=Path("runtime.npz"),
        npy_dir=Path("."),
        cams=["front"],
        out_npy=Path("runtime.npy"),
        hf_repo=args.hf_repo,
        ckpt=args.ckpt,
        mhr_pt=args.mhr_pt,
        device=args.device,
    )
    return load_sam_head(cfg, args.device)


def resolve_faces(head) -> Optional[np.ndarray]:
    faces = getattr(getattr(head, "mhr_model", None), "faces", None)
    if faces is None:
        return None
    arr = np.asarray(faces, dtype=np.int32)
    if arr.size == 0:
        return None
    return arr.reshape(-1, 3)


def sample_coeffs(rng: np.random.Generator, dim: int, std: float, clip: float) -> np.ndarray:
    arr = rng.normal(loc=0.0, scale=float(std), size=(int(dim),)).astype(np.float32)
    if clip > 0:
        np.clip(arr, -float(clip), float(clip), out=arr)
    return arr


def forward_vertices(
    *,
    head,
    device: torch.device,
    body_pose: np.ndarray,
    hand_pose: np.ndarray,
    scale_params: np.ndarray,
    shape_params: np.ndarray,
    expr_params: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        out = mhr_fk(
            head,
            to_torch(body_pose, device).flatten().to(torch.float32),
            to_torch(hand_pose, device).flatten().to(torch.float32),
            to_torch(scale_params, device).flatten().to(torch.float32),
            to_torch(shape_params, device).flatten().to(torch.float32),
            to_torch(expr_params, device).flatten().to(torch.float32),
            device,
            want_verts=True,
            want_joint=False,
            want_model_params=False,
        )
        verts = apply_repo_camera_flip_xyz(out[0].squeeze(0)).cpu().numpy().astype(np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
        raise RuntimeError(f"Unexpected vertex array shape from mhr_forward: {verts.shape}")
    if not np.isfinite(verts).all():
        raise RuntimeError("Generated vertices contain non-finite values.")
    return verts


def main(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Re-run with --device cpu.")
    if int(args.k) <= 0:
        raise ValueError("--k must be positive.")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    lower_body_idxs = resolve_lower_body_pose_indices(int(args.pose_dim))
    device = torch.device(args.device)
    head = build_head(args)
    faces = resolve_faces(head)

    manifest = {
        "k": int(args.k),
        "seed": int(args.seed),
        "device": str(args.device),
        "pose_dim": int(args.pose_dim),
        "body_std": float(args.body_std),
        "hand_std": float(args.hand_std),
        "scale_std": float(args.scale_std),
        "shape_std": float(args.shape_std),
        "expr_std": float(args.expr_std),
        "clip": float(args.clip),
        "lower_body_pose_indices": lower_body_idxs.tolist(),
        "hf_repo": None if args.hf_repo is None else str(args.hf_repo),
        "ckpt": None if args.ckpt is None else str(Path(args.ckpt).expanduser().resolve()),
        "mhr_pt": None if str(args.mhr_pt).strip() == "" else str(Path(args.mhr_pt).expanduser().resolve()),
        "samples": [],
    }

    for sample_idx in range(int(args.k)):
        body_pose = sample_coeffs(rng, int(args.pose_dim), float(args.body_std), float(args.clip))
        body_pose[lower_body_idxs] = 0.0
        hand_pose = sample_coeffs(rng, 108, float(args.hand_std), float(args.clip))
        scale_params = sample_coeffs(rng, 28, float(args.scale_std), float(args.clip))
        shape_params = sample_coeffs(rng, 45, float(args.shape_std), float(args.clip))
        expr_params = sample_coeffs(rng, 72, float(args.expr_std), float(args.clip))

        verts = forward_vertices(
            head=head,
            device=device,
            body_pose=body_pose,
            hand_pose=hand_pose,
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr_params,
        )

        stem = f"sample_{sample_idx:03d}"
        mesh_path = out_dir / f"{stem}.ply"
        params_path = out_dir / f"{stem}.npy"

        write_ply(mesh_path, verts=verts, faces=faces)
        sample_dict = {
            "body_pose_params": body_pose,
            "hand_pose_params": hand_pose,
            "scale_params": scale_params,
            "shape_params": shape_params,
            "expr_params": expr_params,
            "pred_vertices": verts,
            "lower_body_pose_indices": lower_body_idxs.copy(),
            "lower_body_pose_zero": np.array(1, dtype=np.int32),
            "lower_body_pose_abs_max": np.array(
                float(np.max(np.abs(body_pose[lower_body_idxs]))) if lower_body_idxs.size > 0 else 0.0,
                dtype=np.float32,
            ),
            "sample_index": np.array(sample_idx, dtype=np.int32),
            "rng_seed": np.array(int(args.seed), dtype=np.int64),
        }
        if faces is not None:
            sample_dict["faces"] = faces
        save_npy_dict(params_path, sample_dict)
        if bool(args.save_png):
            save_debug_mesh_png(out_dir / f"{stem}.png", verts=verts, title=stem)

        manifest["samples"].append(
            {
                "index": sample_idx,
                "mesh": mesh_path.name,
                "params": params_path.name,
            }
        )
        print(
            f"[{sample_idx + 1}/{int(args.k)}] wrote {mesh_path.name} "
            f"(lower-zero count={int(lower_body_idxs.size)})"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
