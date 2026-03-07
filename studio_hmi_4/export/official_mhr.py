#!/usr/bin/env python3
"""Run official MHR forward from exported compact params.

This script reads per-frame `mhr_params.npz` files (from `export_mhr_final_results.py`)
and regenerates meshes using the official MHR repository API:

  from mhr.mhr import MHR
  mhr_model = MHR.from_files(...)
  verts, skel_state = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)

Per frame, it writes:
  - mesh file (default: `mesh_mhr_repo.ply`)
  - optional debug image (default: `debug_mesh_repo.png`)
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from studio_hmi_4.common import save_debug_mesh_png as _save_debug_mesh, write_ply as _write_ply
from studio_hmi_4.export.common import (
    discover_named_exports,
    extract_optional_scalar_int,
    extract_vec,
    load_params_npz,
)


def _resolve_faces(model: Any) -> Optional[np.ndarray]:
    attr_candidates: List[Tuple[Any, str]] = [
        (model, "faces"),
        (model, "triangles"),
        (model, "face_idx"),
    ]
    nested = getattr(model, "mhr_model", None)
    if nested is not None:
        attr_candidates.extend(
            [
                (nested, "faces"),
                (nested, "triangles"),
                (nested, "face_idx"),
            ]
        )

    for obj, attr in attr_candidates:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            arr = np.asarray(val)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr.astype(np.int32)
    return None

def _build_mhr_model(
    mhr_repo_root: Path,
    assets_dir: Optional[Path],
    device: str,
    lod: int,
):
    sys.path.insert(0, str(mhr_repo_root))
    try:
        from mhr.mhr import MHR
    except Exception as exc:
        raise ImportError(
            f"Failed to import official MHR API from repo root '{mhr_repo_root}'. "
            "Expected module: mhr.mhr"
        ) from exc

    from_files = getattr(MHR, "from_files", None)
    if from_files is None:
        raise AttributeError("MHR.from_files not found in official MHR API.")

    sig = inspect.signature(from_files)
    kwargs: Dict[str, Any] = {}
    if "device" in sig.parameters:
        kwargs["device"] = device
    if "lod" in sig.parameters:
        kwargs["lod"] = int(lod)
    if assets_dir is not None:
        if "folder" in sig.parameters:
            kwargs["folder"] = str(assets_dir)
        elif "assets_dir" in sig.parameters:
            kwargs["assets_dir"] = str(assets_dir)

    model = from_files(**kwargs)
    return model


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run official MHR forward on exported mhr_params.npz")
    ap.add_argument("--export_root", required=True, type=str, help="Root containing per-frame mhr_params.npz files.")
    ap.add_argument(
        "--output_root",
        default="",
        type=str,
        help="Root for generated meshes. Default: same as --export_root.",
    )
    ap.add_argument("--params_name", default="mhr_params.npz", type=str, help="Per-frame params filename.")
    ap.add_argument("--mesh_name", default="mesh_mhr_repo.ply", type=str, help="Per-frame output mesh filename.")
    ap.add_argument(
        "--debug_name",
        default="debug_mesh_repo.png",
        type=str,
        help="Per-frame debug visualization filename.",
    )
    ap.add_argument("--summary_name", default="mhr_repo_forward_summary.json", type=str)
    ap.add_argument("--mhr_repo_root", required=True, type=str, help="Path to cloned official MHR repository root.")
    ap.add_argument(
        "--assets_dir",
        default="",
        type=str,
        help="Optional MHR assets folder. If omitted, MHR.from_files default path is used.",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], type=str)
    ap.add_argument("--lod", default=1, type=int)
    ap.add_argument("--max_frames", default=0, type=int, help="Optional cap (0 = all).")
    ap.add_argument("--skip_bad", action="store_true", default=False, help="Skip frames with is_bad_loss != 0.")
    ap.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing mesh/debug files.")
    ap.add_argument("--debug_vis", action="store_true", default=False, help="Save per-frame debug mesh image.")
    ap.add_argument("--quiet", action="store_true", default=False)
    return ap.parse_args(argv)


def main(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_args()

    export_root = Path(args.export_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else export_root
    mhr_repo_root = Path(args.mhr_repo_root).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve() if args.assets_dir else None
    output_root.mkdir(parents=True, exist_ok=True)

    frames = discover_named_exports(export_root, args.params_name)
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    if not frames:
        print(f"[WARN] No '{args.params_name}' found under {export_root}")
        return 0

    try:
        import torch
    except Exception as exc:
        raise ImportError("PyTorch is required to run official MHR forward.") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        if not bool(args.quiet):
            print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    model = _build_mhr_model(
        mhr_repo_root=mhr_repo_root,
        assets_dir=assets_dir,
        device=device,
        lod=int(args.lod),
    )
    faces = _resolve_faces(model)

    num_total = 0
    num_exported = 0
    num_skipped = 0
    num_bad = 0
    num_errors = 0

    for rec in frames:
        num_total += 1
        try:
            d = load_params_npz(rec.path)
            is_bad = extract_optional_scalar_int(d, "is_bad_loss", default=0)
            if is_bad:
                num_bad += 1
                if bool(args.skip_bad):
                    num_skipped += 1
                    continue

            frame_out = (output_root / rec.rel_dir).resolve()
            frame_out.mkdir(parents=True, exist_ok=True)
            mesh_path = frame_out / str(args.mesh_name)
            debug_path = frame_out / str(args.debug_name)

            if mesh_path.exists() and (not bool(args.overwrite)):
                if not bool(args.debug_vis) or debug_path.exists():
                    num_skipped += 1
                    continue

            mhr_model_params = extract_vec(d, "mhr_model_params", dim=204)
            shape_params = extract_vec(d, "shape_params", dim=45)
            expr_params = extract_vec(d, "expr_params", dim=72)

            identity_coeffs = torch.from_numpy(shape_params).to(device=device, dtype=torch.float32).view(1, -1)
            model_parameters = torch.from_numpy(mhr_model_params).to(device=device, dtype=torch.float32).view(1, -1)
            face_expr_coeffs = torch.from_numpy(expr_params).to(device=device, dtype=torch.float32).view(1, -1)

            with torch.no_grad():
                out = model(identity_coeffs, model_parameters, face_expr_coeffs)

            verts_out = out[0] if isinstance(out, (tuple, list)) else out
            verts = np.asarray(verts_out.detach().cpu().numpy(), dtype=np.float32)
            if verts.ndim == 3 and verts.shape[0] == 1:
                verts = verts[0]
            if verts.ndim != 2 or verts.shape[1] != 3:
                raise ValueError(f"Unexpected verts shape from MHR forward: {verts.shape}")
            if not np.isfinite(verts).all():
                raise ValueError("MHR output vertices contain non-finite values")

            _write_ply(mesh_path, verts=verts, faces=faces)
            if bool(args.debug_vis):
                _save_debug_mesh(debug_path, verts=verts, title=f"frame={rec.rel_dir}")
            num_exported += 1
        except Exception as exc:
            num_errors += 1
            if not bool(args.quiet):
                print(f"[ERROR] frame '{rec.rel_dir}': {type(exc).__name__}: {exc}")

    summary = {
        "export_root": str(export_root),
        "output_root": str(output_root),
        "params_name": str(args.params_name),
        "mesh_name": str(args.mesh_name),
        "mhr_repo_root": str(mhr_repo_root),
        "assets_dir": "" if assets_dir is None else str(assets_dir),
        "device": str(device),
        "lod": int(args.lod),
        "skip_bad": bool(args.skip_bad),
        "debug_vis": bool(args.debug_vis),
        "num_total": int(num_total),
        "num_exported": int(num_exported),
        "num_skipped": int(num_skipped),
        "num_bad_inputs": int(num_bad),
        "num_errors": int(num_errors),
    }
    summary_path = (output_root / str(args.summary_name)).resolve()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[run_mhr_repo_from_export] total={num_total} exported={num_exported} "
        f"skipped={num_skipped} bad_inputs={num_bad} errors={num_errors} out={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
