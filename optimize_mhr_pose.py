#!/usr/bin/env python3
from __future__ import annotations
"""
Optimize sam-3d-body body_pose_params (133-D) using GT 3D keypoints from your refined multi-view triangulation.

Inputs:
  - a directory of SAM outputs .npy (dict saved by your sam-3d-body inference per view)
      expected: <npy_dir>/<cam>.npy  for cam in --cams
      (fallback: unique glob match *<cam>*.npy)
  - 1x triangulation/refinement .npz:
      points3d_refined (M,3), subset_indices (M,), subset_names (M,), inlier_mask (M,V),
      <cam>_mean_err_px_refined ...

What it does:
  1) Scores each view by 3D Procrustes (Umeyama) residual on the subset points.
  2) Picks best view as init (3D score primary, mean px as tie-break).
  3) Alternating optimize:
     - FK -> canonical 70 keypoints
     - Umeyama (s,R,t) to GT subset (SVD, no grad)
     - gradient step on ONE shared body_pose (133-D)

Outputs:
  - out_npy: same dict keys as init view npy, but updated:
      body_pose_params, pred_keypoints_3d, pred_vertices, pred_joint_coords, pred_global_rots, mhr_model_params
    plus extra debug keys prefixed by opt_*
  - debug_dir: debug_opt.npz + plots

Example:
  python optimize_mhr_pose.py \
    --npz refined_points.npz \
    --npy_dir /path/to/sam_outputs \
    --cams left front right \
    --hf_repo facebook/sam-3d-body-dinov3 \
    --with_scale --iters 200 --lr 0.05 \
    --debug_dir debug_opt \
    --out_npy opt_out.npy
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch

DEFAULT_PARAM_DIMS = {
    "hand_pose_params": 108,
    "scale_params": 28,
    "shape_params": 45,
    "expr_params": 72,
}


@dataclass
class OptimizationConfig:
    npz: Path
    npy_dir: Path
    cams: List[str]
    out_npy: Path
    debug_dir: Path = Path("debug_opt")
    hf_repo: Optional[str] = None
    ckpt: Optional[str] = None
    mhr_pt: str = ""
    device: str = "cuda"
    iters: int = 200
    lr: float = 5e-2
    with_scale: bool = False
    huber_m: float = 0.03
    w_pose_reg: float = 1e-3
    topk_print: int = 10


@dataclass
class OptimizationRunResult:
    out_npy: Path
    debug_dir: Path
    best_cam: str
    loss_history: List[float]


# -----------------------------
# Weighted Umeyama similarity
# -----------------------------
@torch.no_grad()
def umeyama_similarity(X, Y, w=None, with_scale=True, eps=1e-9):
    """
    Solve: Y ≈ s * (X @ R^T) + t
    X, Y: (N,3) tensors
    w: (N,) weights in [0,1] (optional)
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    N = X.shape[0]
    if w is None:
        w = torch.ones(N, device=X.device, dtype=X.dtype)
    w = w.clamp(min=0)
    wsum = w.sum().clamp(min=eps)
    w = w / wsum

    muX = (w[:, None] * X).sum(0)
    muY = (w[:, None] * Y).sum(0)
    Xc = X - muX
    Yc = Y - muY

    S = (Xc * w[:, None]).T @ Yc  # (3,3)
    U, D, Vt = torch.linalg.svd(S)
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if with_scale:
        varX = (w[:, None] * (Xc * Xc)).sum()
        s = D.sum() / varX.clamp(min=eps)
    else:
        s = torch.tensor(1.0, device=X.device, dtype=X.dtype)

    t = muY - s * (muX @ R.T)
    return s, R, t


def huber(r, delta):
    # r: (...) nonnegative
    return torch.where(r <= delta, 0.5 * r * r, delta * (r - 0.5 * delta))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_torch(x, device, dtype=torch.float32):
    return torch.tensor(x, device=device, dtype=dtype)


def safe_numpy_item_load(npy_path: Path):
    d = np.load(npy_path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == () and hasattr(d, "item"):
        return d.item()
    if isinstance(d, dict):
        return d
    raise ValueError(f"Unexpected npy content at {npy_path}: {type(d)}")


def find_npy_for_cam(npy_dir: Path, cam: str) -> Path:
    direct = npy_dir / f"{cam}.npy"
    if direct.exists():
        return direct
    matches = sorted(npy_dir.glob(f"*{cam}*.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find {cam}.npy or *{cam}*.npy in {npy_dir}")
    raise RuntimeError(f"Ambiguous matches for cam='{cam}' in {npy_dir}: {[m.name for m in matches]}")


def get_param_array(d: dict, key: str, device: torch.device) -> torch.Tensor:
    if key in d:
        return to_torch(d[key], device).flatten()

    dim = DEFAULT_PARAM_DIMS[key]
    return to_torch(np.zeros(dim, np.float32), device).flatten()


# -----------------------------
# sam-3d-body forward FK helper
# -----------------------------
def load_sam_head(config: OptimizationConfig, device_str: str):
    from sam_3d_body.build_models import load_sam_3d_body, load_sam_3d_body_hf

    if config.hf_repo:
        model, _cfg = load_sam_3d_body_hf(config.hf_repo, device=device_str)
    else:
        model, _cfg = load_sam_3d_body(
            checkpoint_path=config.ckpt, device=device_str, mhr_path=config.mhr_pt
        )
    head = model.head_pose
    head.eval()
    return head


def apply_repo_camera_flip_xyz(x):
    """
    sam-3d-body flips y,z after MHR in MHRHead.forward for verts/keypoints/joints.
    Keep consistent with your saved pred_keypoints_3d.
    """
    x = x.clone()
    x[..., 1] *= -1.0
    x[..., 2] *= -1.0
    return x


def mhr_fk(
    head,
    body_pose_eff_133,
    hand_pose_108,
    scale_28,
    shape_45,
    expr_72,
    device,
    want_verts=True,
    want_joint=True,
    want_model_params=True,
):
    """
    Calls head.mhr_forward to get (verts, keypoints, joint_coords, model_params, joint_rots).
    We try passing pose[:130] first, then fall back to full pose if needed.
    """
    B = 1
    global_trans = torch.zeros(B, 3, device=device)
    global_rot = torch.zeros(B, 3, device=device)

    pose133 = body_pose_eff_133.view(B, -1)
    hand108 = hand_pose_108.view(B, -1)
    scale28 = scale_28.view(B, -1)
    shape45 = shape_45.view(B, -1)
    expr72 = expr_72.view(B, -1)

    pose_try_list = [pose133[:, :130], pose133]

    last_err = None
    for pose_try in pose_try_list:
        try:
            out = head.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot,
                body_pose_params=pose_try,
                hand_pose_params=hand108,
                scale_params=scale28,
                shape_params=shape45,
                expr_params=expr72,
                return_keypoints=True,
                return_joint_coords=want_joint,
                return_model_params=want_model_params,
                return_joint_rotations=want_joint,
            )
            if not isinstance(out, tuple):
                raise RuntimeError("Unexpected mhr_forward return type.")
            return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"mhr_forward failed for both pose sizes. Last error: {last_err}")


# -----------------------------
# Debug plotting
# -----------------------------
def set_axes_equal(ax):
    import numpy as _np
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = _np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = _np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = _np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3d_compare(gtM, predM, title, out_png: Path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    gtM = np.asarray(gtM)
    predM = np.asarray(predM)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gtM[:, 0], gtM[:, 1], gtM[:, 2], s=30, marker="o", label="GT refined")
    ax.scatter(predM[:, 0], predM[:, 1], predM[:, 2], s=30, marker="^", label="Pred aligned")

    for i in range(gtM.shape[0]):
        ax.plot([gtM[i, 0], predM[i, 0]],
                [gtM[i, 1], predM[i, 1]],
                [gtM[i, 2], predM[i, 2]], linewidth=0.5)

    ax.set_title(title)
    ax.legend()
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_loss_curve(loss_hist, out_png: Path):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 4))
    plt.plot(loss_hist)
    plt.title("Optimization loss")
    plt.xlabel("iter")
    plt.ylabel("loss")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray = None):
    verts = np.asarray(verts).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if faces is not None:
            faces = np.asarray(faces).reshape(-1, 3)
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        if faces is not None:
            for tri in faces:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


# -----------------------------
# Main
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--npy_dir", required=True, type=Path)
    ap.add_argument("--cams", nargs="+", required=True, help="Camera names, e.g. left front right")
    ap.add_argument("--out_npy", required=True, type=Path)
    ap.add_argument("--debug_dir", default=Path("debug_opt"), type=Path)

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hf_repo", type=str, default=None)
    grp.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mhr_pt", type=str, default="", help="needed if using --ckpt")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--with_scale", action="store_true")
    ap.add_argument("--huber_m", type=float, default=0.03, help="Huber delta in meters")
    ap.add_argument("--w_pose_reg", type=float, default=1e-3)
    ap.add_argument("--topk_print", type=int, default=10)
    return ap


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> OptimizationConfig:
    return OptimizationConfig(
        npz=args.npz,
        npy_dir=args.npy_dir,
        cams=list(args.cams),
        out_npy=args.out_npy,
        debug_dir=args.debug_dir,
        hf_repo=args.hf_repo,
        ckpt=args.ckpt,
        mhr_pt=args.mhr_pt,
        device=args.device,
        iters=args.iters,
        lr=args.lr,
        with_scale=args.with_scale,
        huber_m=args.huber_m,
        w_pose_reg=args.w_pose_reg,
        topk_print=args.topk_print,
    )


def run_optimization(config: OptimizationConfig) -> OptimizationRunResult:
    if not config.hf_repo and not config.ckpt:
        raise ValueError("Either hf_repo or ckpt must be provided.")

    device = torch.device(config.device)
    debug_dir = ensure_dir(config.debug_dir.expanduser().resolve())
    out_npy = config.out_npy.expanduser().resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    cams = list(config.cams)
    npy_dir = config.npy_dir.expanduser().resolve()
    npz_path = config.npz.expanduser().resolve()

    # ---- Load refined GT npz ----
    z = np.load(npz_path, allow_pickle=True)
    subset_idx = z["subset_indices"].astype(np.int64).reshape(-1)   # (M,)
    subset_names = z.get("subset_names", None)
    gtM = z["points3d_refined"].astype(np.float32)                 # (M,3)
    M = gtM.shape[0]

    # weights from inlier mask (M,V) -> mean confidence per point
    if "inlier_mask" in z:
        wM = z["inlier_mask"].astype(np.float32)
        if wM.ndim == 2:
            wM = wM.mean(axis=1)
        wM = np.clip(wM, 0.0, 1.0)
    else:
        wM = np.ones((M,), dtype=np.float32)

    gtM_t = to_torch(gtM, device)
    wM_t = to_torch(wM, device)

    # ---- Load sam-3d-body head ----
    head = load_sam_head(config, config.device)

    # ---- Build keep-mask for pose (fix autograd issue) ----
    from sam_3d_body.models.modules.mhr_utils import mhr_param_hand_mask
    hand_mask = mhr_param_hand_mask.to(device)  # bool (133,)
    keep_mask = torch.ones(133, device=device, dtype=torch.float32)
    keep_mask[hand_mask] = 0.0
    keep_mask[-3:] = 0.0  # jaw

    # ---- Load per-view SAM outputs and score them ----
    view_scores_3d = {}
    view_mean_px = {}
    view_align = {}
    view_dicts = {}

    # mean px keys come from your npz: <cam>_mean_err_px_refined
    for cam in cams:
        k = f"{cam}_mean_err_px_refined"
        view_mean_px[cam] = float(np.asarray(z[k]).reshape(())) if k in z else float("inf")

    for cam in cams:
        npy_path = find_npy_for_cam(npy_dir, cam)
        d = safe_numpy_item_load(npy_path)
        view_dicts[cam] = d

        pose133 = to_torch(d["body_pose_params"], device).flatten().to(torch.float32)
        hand108 = get_param_array(d, "hand_pose_params", device)
        scale28 = get_param_array(d, "scale_params", device)
        shape45 = get_param_array(d, "shape_params", device)
        expr72 = get_param_array(d, "expr_params", device)

        # mask (NO in-place on leaf; this one isn't leaf anyway but keep consistent)
        pose_eff = pose133 * keep_mask

        out = mhr_fk(head, pose_eff, hand108, scale28, shape45, expr72, device,
                     want_verts=False, want_joint=False, want_model_params=False)
        keypoints_308 = out[1].squeeze(0)   # (308,3)
        k70 = apply_repo_camera_flip_xyz(keypoints_308[:70])

        predM = k70[subset_idx]  # (M,3)

        s, R, t = umeyama_similarity(predM, gtM_t, w=wM_t, with_scale=config.with_scale)
        predM_aligned = s * (predM @ R.T) + t[None, :]
        r = torch.sqrt(((predM_aligned - gtM_t) ** 2).sum(dim=1) + 1e-12)
        score = (wM_t * r).sum() / (wM_t.sum() + 1e-9)

        view_scores_3d[cam] = float(score.detach().cpu().item())
        view_align[cam] = (float(s.cpu().item()), R.cpu().numpy(), t.cpu().numpy())

        print(f"[score] {cam:>8s}: 3D={view_scores_3d[cam]:.6f} m   mean_px={view_mean_px[cam]:.3f}   file={npy_path.name}")

    best_cam = sorted(cams, key=lambda c: (view_scores_3d[c], view_mean_px[c]))[0]
    print(f"\n[init] best_cam = {best_cam}  (3D={view_scores_3d[best_cam]:.6f} m, mean_px={view_mean_px[best_cam]:.3f})")

    init_dict = view_dicts[best_cam]

    init_pose = to_torch(init_dict["body_pose_params"], device).flatten().to(torch.float32)
    init_hand = get_param_array(init_dict, "hand_pose_params", device)
    init_scale = get_param_array(init_dict, "scale_params", device)
    init_shape = get_param_array(init_dict, "shape_params", device)
    init_expr = get_param_array(init_dict, "expr_params", device)

    # Mask init pose (no in-place leaf issue here)
    init_pose = init_pose * keep_mask

    # ---- Optimize pose (leaf tensor) ----
    pose = init_pose.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pose], lr=config.lr)

    loss_hist = []

    # debug init plot
    with torch.no_grad():
        pose_eff0 = pose.detach() * keep_mask
        out0 = mhr_fk(head, pose_eff0, init_hand, init_scale, init_shape, init_expr, device,
                      want_verts=False, want_joint=False, want_model_params=False)
        k70_0 = apply_repo_camera_flip_xyz(out0[1].squeeze(0)[:70])
        predM0 = k70_0[subset_idx]
        s0, R0, t0 = umeyama_similarity(predM0, gtM_t, w=wM_t, with_scale=config.with_scale)
        predM0_al = s0 * (predM0 @ R0.T) + t0[None, :]
        plot_3d_compare(gtM, predM0_al.cpu().numpy(),
                        f"Init ({best_cam}) aligned to GT (M={M})", debug_dir / "compare_init_subset.png")

    # main loop
    for it in range(config.iters):
        opt.zero_grad(set_to_none=True)

        # effective pose used in FK (NO in-place on pose)
        pose_eff = pose * keep_mask

        out = mhr_fk(head, pose_eff, init_hand, init_scale, init_shape, init_expr, device,
                     want_verts=False, want_joint=False, want_model_params=False)
        k70 = apply_repo_camera_flip_xyz(out[1].squeeze(0)[:70])
        predM = k70[subset_idx]

        with torch.no_grad():
            s, R, t = umeyama_similarity(predM.detach(), gtM_t, w=wM_t, with_scale=config.with_scale)

        predM_aligned = s * (predM @ R.T) + t[None, :]
        diff = predM_aligned - gtM_t
        r = torch.sqrt((diff * diff).sum(dim=1) + 1e-12)

        loss_data = (wM_t * huber(r, delta=config.huber_m)).sum() / (wM_t.sum() + 1e-9)
        loss_reg = config.w_pose_reg * torch.mean((pose - init_pose) ** 2)
        loss = loss_data + loss_reg

        loss.backward()

        # zero grads on frozen dims (important)
        with torch.no_grad():
            if pose.grad is not None:
                pose.grad[hand_mask] = 0.0
                pose.grad[-3:] = 0.0

        torch.nn.utils.clip_grad_norm_([pose], 1.0)
        opt.step()

        # pin values (optional but keeps optimizer state sane)
        with torch.no_grad():
            pose.mul_(keep_mask)

        loss_hist.append(float(loss.detach().cpu().item()))
        if it % 25 == 0 or it == config.iters - 1:
            print(f"[{it:04d}] loss={loss_hist[-1]:.6f} data={float(loss_data):.6f}")

    plot_loss_curve(loss_hist, debug_dir / "loss_curve.png")

    # final forward (full outputs)
    with torch.no_grad():
        pose_eff = pose.detach() * keep_mask
        outF = mhr_fk(head, pose_eff, init_hand, init_scale, init_shape, init_expr, device,
                      want_verts=True, want_joint=True, want_model_params=True)

        # expected order: verts, keypoints, joint_coords, model_params, joint_rots
        verts = apply_repo_camera_flip_xyz(outF[0].squeeze(0))
        keypoints_308 = outF[1].squeeze(0)
        jcoords = apply_repo_camera_flip_xyz(outF[2].squeeze(0))
        model_params = outF[3].squeeze(0)
        jrots = outF[4].squeeze(0)

        k70 = apply_repo_camera_flip_xyz(keypoints_308[:70])
        predM = k70[subset_idx]
        s, R, t = umeyama_similarity(predM, gtM_t, w=wM_t, with_scale=config.with_scale)

        k70_aligned = s * (k70 @ R.T) + t[None, :]
        verts_aligned = s * (verts @ R.T) + t[None, :]
        jcoords_aligned = s * (jcoords @ R.T) + t[None, :]
        jrots_aligned = R[None, :, :] @ jrots  # optional consistency

        rM = torch.sqrt(((k70_aligned[subset_idx] - gtM_t) ** 2).sum(dim=1) + 1e-12).cpu().numpy()
        worst = np.argsort(-rM)[: config.topk_print]
        print("\nTop residual points (subset indices):")
        for idx in worst:
            name = str(subset_names[idx]) if subset_names is not None else f"pt{idx}"
            print(f"  {idx:02d} ({name}): {rM[idx]:.4f} m")

        plot_3d_compare(gtM, k70_aligned[subset_idx].cpu().numpy(),
                        f"Optimized aligned to GT (M={M})", debug_dir / "compare_opt_subset.png")

        faces = init_dict.get("faces", None)
        if faces is not None:
            write_ply(debug_dir / "mesh_opt_aligned.ply", verts_aligned.cpu().numpy(), faces)
        else:
            write_ply(debug_dir / "verts_opt_aligned.ply", verts_aligned.cpu().numpy(), None)

        np.savez_compressed(
            debug_dir / "debug_opt.npz",
            best_cam=np.array(best_cam, dtype=object),
            cams=np.array(cams, dtype=object),
            subset_idx=subset_idx,
            gt_subset=gtM,
            w_subset=wM,
            init_scores_3d_m=np.array([view_scores_3d[c] for c in cams], dtype=np.float32),
            init_scores_mean_px=np.array([view_mean_px[c] for c in cams], dtype=np.float32),
            final_scale=np.array(float(s.cpu().item()), dtype=np.float32),
            final_R=R.cpu().numpy(),
            final_t=t.cpu().numpy(),
            loss_hist=np.array(loss_hist, dtype=np.float32),
            residual_subset_m=np.array(rM, dtype=np.float32),
            keep_mask=keep_mask.cpu().numpy(),
        )

        # output npy dict in same style as SAM output
        out_dict = dict(init_dict)
        out_dict["body_pose_params"] = (pose_eff.cpu().numpy()).astype(np.float32)
        out_dict["pred_keypoints_3d"] = k70_aligned.cpu().numpy().astype(np.float32)
        out_dict["pred_vertices"] = verts_aligned.cpu().numpy().astype(np.float32)
        out_dict["pred_joint_coords"] = jcoords_aligned.cpu().numpy().astype(np.float32)
        out_dict["pred_global_rots"] = jrots_aligned.cpu().numpy().astype(np.float32)
        out_dict["mhr_model_params"] = model_params.cpu().numpy().astype(np.float32)

        # debug extras
        out_dict["opt_init_cam"] = best_cam
        out_dict["opt_cam_scores_3d_m"] = view_scores_3d
        out_dict["opt_cam_mean_err_px_refined"] = view_mean_px
        out_dict["opt_sim_scale"] = float(s.cpu().item())
        out_dict["opt_sim_R"] = R.cpu().numpy()
        out_dict["opt_sim_t"] = t.cpu().numpy()
        out_dict["opt_loss_hist"] = np.array(loss_hist, dtype=np.float32)
        out_dict["opt_subset_indices"] = subset_idx
        out_dict["opt_points3d_refined"] = gtM

        np.save(out_npy, out_dict, allow_pickle=True)

    print(f"\nSaved optimized npy: {out_npy}")
    print(f"Saved debug dir: {debug_dir}")
    return OptimizationRunResult(
        out_npy=out_npy,
        debug_dir=debug_dir,
        best_cam=best_cam,
        loss_history=loss_hist,
    )


def main(args: Optional[argparse.Namespace] = None) -> OptimizationRunResult:
    if args is None:
        args = parse_args()
    config = namespace_to_config(args)
    return run_optimization(config)


if __name__ == "__main__":
    main()
