#!/usr/bin/env python3
"""Export compact MHR-ready per-frame results from pipeline optimization outputs.

This script reads per-frame optimization outputs (e.g., `opt_out_smoothed.npy` or
`opt_out.npy`) and writes:
  - one compact `mhr_params.npy` per frame (dict)
  - one compact `mhr_params.npz` per frame (arrays)
  - one `mesh.ply` per frame (optional)
  - one `debug_mesh.png` per frame (optional debug visualization)

Output frame directory keeps the same relative frame folder name as optimization
input (for example `0/`, `1/`, ...).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from studio_hmi_4.common import (
    load_npy_dict,
    save_debug_mesh_png as _save_debug_mesh_png,
    validate_optimization_result_dict,
    write_ply,
)
from studio_hmi_4.export.common import discover_frame_exports, extract_optional_vec


def _as_vertices(data: Dict[str, Any], key: str = "pred_vertices") -> Optional[np.ndarray]:
    if key not in data:
        return None
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr

class MHRForwardRunner:
    """Lazy loader for SAM head.mhr_forward-based mesh regeneration."""

    def __init__(
        self,
        hf_repo: Optional[str],
        ckpt: Optional[str],
        mhr_pt: str,
        device: str,
    ):
        self.device_name = str(device)
        self.hf_repo = hf_repo
        self.ckpt = ckpt
        self.mhr_pt = str(mhr_pt)
        self._loaded = False
        self._device = None
        self._head = None
        self._faces = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from studio_hmi_4.stage3.runner import (
            OptimizationConfig,
            apply_repo_camera_flip_xyz,
            load_sam_head,
            mhr_fk,
            to_torch,
        )

        self._torch = torch
        self._apply_repo_camera_flip_xyz = apply_repo_camera_flip_xyz
        self._mhr_fk = mhr_fk
        self._to_torch = to_torch

        device = torch.device(self.device_name)
        cfg = OptimizationConfig(
            npz=Path("runtime.npz"),
            npy_dir=Path("."),
            cams=["front"],
            out_npy=Path("runtime.npy"),
            hf_repo=self.hf_repo,
            ckpt=self.ckpt,
            mhr_pt=self.mhr_pt,
            device=self.device_name,
        )
        head = load_sam_head(cfg, self.device_name)
        self._device = device
        self._head = head
        faces = getattr(getattr(head, "mhr_model", None), "faces", None)
        self._faces = None if faces is None else np.asarray(faces, dtype=np.int32)
        self._loaded = True

    def get_faces(self) -> Optional[np.ndarray]:
        self._ensure_loaded()
        return self._faces

    def forward_vertices(
        self,
        body_pose_133: np.ndarray,
        hand_108: np.ndarray,
        scale_28: np.ndarray,
        shape_45: np.ndarray,
        expr_72: np.ndarray,
    ) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        with torch.no_grad():
            pose = self._to_torch(body_pose_133, self._device).flatten().to(torch.float32)
            hand = self._to_torch(hand_108, self._device).flatten().to(torch.float32)
            scale = self._to_torch(scale_28, self._device).flatten().to(torch.float32)
            shape = self._to_torch(shape_45, self._device).flatten().to(torch.float32)
            expr = self._to_torch(expr_72, self._device).flatten().to(torch.float32)
            out = self._mhr_fk(
                self._head,
                pose,
                hand,
                scale,
                shape,
                expr,
                self._device,
                want_verts=True,
                want_joint=False,
                want_model_params=False,
            )
            verts = self._apply_repo_camera_flip_xyz(out[0].squeeze(0)).cpu().numpy().astype(np.float32)
        return verts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("Export compact MHR-ready per-frame results from pipeline outputs")
    ap.add_argument("--optimization_root", required=True, type=str, help="Path to pipeline optimization root.")
    ap.add_argument("--output_root", required=True, type=str, help="Path to exported final-result root.")
    ap.add_argument("--npy_name", type=str, default="opt_out_smoothed.npy", help="Preferred per-frame npy file name.")
    ap.add_argument("--fallback_npy_name", type=str, default="opt_out.npy", help="Fallback per-frame npy file name.")
    ap.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing exported files.")
    ap.add_argument("--max_frames", type=int, default=0, help="Optional limit of exported frames (0 = all).")
    ap.add_argument("--skip_bad", dest="skip_bad", action="store_true", default=True, help="Skip frames marked bad optimization.")
    ap.add_argument("--keep_bad", dest="skip_bad", action="store_false", help="Keep/export frames marked bad optimization.")
    ap.add_argument(
        "--no_decomposed_params",
        action="store_true",
        default=False,
        help="Export only compact fields (mhr_model_params, shape_params, expr_params).",
    )
    ap.add_argument(
        "--no_run_mhr_forward",
        action="store_true",
        default=False,
        help="Do not run MHR forward; export params only (mesh from pred_vertices if available).",
    )
    ap.add_argument("--hf_repo", type=str, default=None, help="HF repo for model load (unless --no_run_mhr_forward).")
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (alternative to --hf_repo).")
    ap.add_argument("--mhr_pt", type=str, default="", help="MHR model path (used with --ckpt).")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for MHR forward.")
    ap.add_argument("--debug_vis", action="store_true", default=False, help="Save per-frame debug mesh preview image.")
    ap.add_argument("--mesh_name", type=str, default="mesh.ply", help="Output mesh filename per frame.")
    ap.add_argument("--quiet", action="store_true", default=False)
    return ap.parse_args(argv)


def main(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_args()

    optimization_root = Path(args.optimization_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = discover_frame_exports(
        root=optimization_root,
        preferred_name=str(args.npy_name),
        fallback_name=str(args.fallback_npy_name),
    )
    if int(args.max_frames) > 0:
        records = records[: int(args.max_frames)]
    if len(records) == 0:
        print(f"[WARN] No per-frame outputs found in: {optimization_root}")
        return 0

    need_forward = not bool(args.no_run_mhr_forward)
    forward_runner: Optional[MHRForwardRunner] = None
    if need_forward:
        if not args.hf_repo and not args.ckpt:
            raise ValueError("When --run_mhr_forward is enabled, provide --hf_repo or --ckpt.")
        forward_runner = MHRForwardRunner(
            hf_repo=args.hf_repo,
            ckpt=args.ckpt,
            mhr_pt=str(args.mhr_pt),
            device=str(args.device),
        )

    num_total = 0
    num_exported = 0
    num_skipped = 0
    num_errors = 0
    num_bad = 0

    for rec in records:
        num_total += 1
        try:
            d = load_npy_dict(rec.path)
            validate_optimization_result_dict(
                d,
                require_geometry=False,
                require_mhr_compact=False,
            )
            is_bad = bool(int(np.asarray(d.get("opt_is_bad_loss", 0)).reshape(())))
            if is_bad:
                num_bad += 1
                if bool(args.skip_bad):
                    num_skipped += 1
                    continue

            frame_out_dir = (output_root / rec.rel_dir).resolve()
            frame_out_dir.mkdir(parents=True, exist_ok=True)
            compact_npy = frame_out_dir / "mhr_params.npy"
            compact_npz = frame_out_dir / "mhr_params.npz"
            mesh_path = frame_out_dir / str(args.mesh_name)
            debug_png = frame_out_dir / "debug_mesh.png"

            if (
                compact_npy.exists()
                and compact_npz.exists()
                and ((not need_forward) or mesh_path.exists())
                and (not bool(args.overwrite))
            ):
                num_skipped += 1
                continue

            # Required compact fields.
            mhr_model_params = extract_optional_vec(d, "mhr_model_params", expected_dim=204)
            shape_params = extract_optional_vec(d, "shape_params", expected_dim=45)
            expr_params = extract_optional_vec(d, "expr_params", expected_dim=72)
            if mhr_model_params is None or shape_params is None or expr_params is None:
                raise KeyError(
                    "Missing required MHR compact fields: mhr_model_params(204), "
                    "shape_params(45), expr_params(72)."
                )

            body_pose = extract_optional_vec(d, "body_pose_params", expected_dim=133)
            hand_pose = extract_optional_vec(d, "hand_pose_params", expected_dim=108)
            scale_params = extract_optional_vec(d, "scale_params", expected_dim=28)

            compact_dict: Dict[str, Any] = {
                "frame_rel": rec.rel_dir,
                "frame_index": -1 if rec.frame_index is None else int(rec.frame_index),
                "source_npy": str(rec.path),
                "is_bad_loss": int(is_bad),
                "mhr_model_params": mhr_model_params.astype(np.float32),
                "shape_params": shape_params.astype(np.float32),
                "expr_params": expr_params.astype(np.float32),
            }
            include_decomposed = not bool(args.no_decomposed_params)
            if include_decomposed:
                if body_pose is not None:
                    compact_dict["body_pose_params"] = body_pose.astype(np.float32)
                if hand_pose is not None:
                    compact_dict["hand_pose_params"] = hand_pose.astype(np.float32)
                if scale_params is not None:
                    compact_dict["scale_params"] = scale_params.astype(np.float32)

            np.save(compact_npy, compact_dict, allow_pickle=True)

            npz_kwargs: Dict[str, Any] = {
                "mhr_model_params": mhr_model_params.astype(np.float32),
                "shape_params": shape_params.astype(np.float32),
                "expr_params": expr_params.astype(np.float32),
                "is_bad_loss": np.array(int(is_bad), dtype=np.int32),
            }
            if include_decomposed:
                if body_pose is not None:
                    npz_kwargs["body_pose_params"] = body_pose.astype(np.float32)
                if hand_pose is not None:
                    npz_kwargs["hand_pose_params"] = hand_pose.astype(np.float32)
                if scale_params is not None:
                    npz_kwargs["scale_params"] = scale_params.astype(np.float32)
            np.savez_compressed(compact_npz, **npz_kwargs)

            verts = None
            faces = None
            if need_forward:
                if body_pose is None or hand_pose is None or scale_params is None:
                    raise KeyError(
                        "MHR forward requires body_pose_params(133), hand_pose_params(108), "
                        "scale_params(28), shape_params(45), expr_params(72)."
                    )
                assert forward_runner is not None
                verts = forward_runner.forward_vertices(
                    body_pose_133=body_pose,
                    hand_108=hand_pose,
                    scale_28=scale_params,
                    shape_45=shape_params,
                    expr_72=expr_params,
                )
                faces = forward_runner.get_faces()
            else:
                verts = _as_vertices(d, key="pred_vertices")

            if verts is not None:
                write_ply(mesh_path, verts=verts, faces=faces)
                if bool(args.debug_vis):
                    _save_debug_mesh_png(
                        path=debug_png,
                        verts=verts,
                        title=f"frame={rec.rel_dir}",
                    )
            elif not bool(args.quiet):
                print(f"[WARN] No vertices available for frame '{rec.rel_dir}'")

            num_exported += 1
        except Exception as exc:
            num_errors += 1
            if not bool(args.quiet):
                print(f"[ERROR] frame '{rec.rel_dir}': {type(exc).__name__}: {exc}")

    summary = {
        "optimization_root": str(optimization_root),
        "output_root": str(output_root),
        "npy_name": str(args.npy_name),
        "fallback_npy_name": str(args.fallback_npy_name),
        "run_mhr_forward": bool(need_forward),
        "hf_repo": args.hf_repo,
        "ckpt": args.ckpt,
        "mhr_pt": str(args.mhr_pt),
        "device": str(args.device),
        "skip_bad": bool(args.skip_bad),
        "include_decomposed_params": not bool(args.no_decomposed_params),
        "num_total": int(num_total),
        "num_exported": int(num_exported),
        "num_skipped": int(num_skipped),
        "num_bad_inputs": int(num_bad),
        "num_errors": int(num_errors),
    }
    summary_path = (output_root / "export_summary.json").resolve()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[export_mhr_final_results] total={num_total} exported={num_exported} "
        f"skipped={num_skipped} bad_inputs={num_bad} errors={num_errors} "
        f"out={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
