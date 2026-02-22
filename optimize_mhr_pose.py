#!/usr/bin/env python3
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
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import torch

DEFAULT_PARAM_DIMS = {
    "hand_pose_params": 108,
    "scale_params": 28,
    "shape_params": 45,
    "expr_params": 72,
}

# Source for hand indices: upstream sam-3d-body `mhr_utils.py` (`mhr_param_hand_idxs`).
MHR_PARAM_HAND_IDXS_133 = np.array(
    [
        62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115,
    ],
    dtype=np.int64,
)

# Source for lower-body semantics: local `mhr70.py` lower keypoint groups + one-time
# audit of MHR body parameter layout from upstream `mhr_utils.py`/`mhr_head.py`.
LOWER_BODY_POSE_IDXS_BY_DIM = {
    # 133D body pose params:
    # - 0..30 cover pelvis/leg chains in this parameterization.
    # - 124..129 are translational DoFs and are frozen to suppress standing jitter.
    133: np.array(list(range(0, 31)) + [124, 125, 126, 127, 128, 129], dtype=np.int64),
    # 204D full MHR model params:
    # - first 6 are rigid global params
    # - body pose block starts at +6 offset from the 133D pose indexing
    204: np.array(
        sorted(
            set(
                list(range(0, 6))
                + [6 + i for i in range(0, 31)]
                + [6 + i for i in [124, 125, 126, 127, 128, 129]]
            )
        ),
        dtype=np.int64,
    ),
}


@dataclass
class OptimizationConfig:
    """Configuration for stage-3 pose optimization against triangulated 3D GT."""

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
    save_debug_artifacts: bool = True
    min_iters: int = 50
    early_stop_patience: int = 60
    early_stop_tol: float = 1e-6
    init_body_pose: Optional[np.ndarray] = None
    init_prev_body_pose: Optional[np.ndarray] = None
    init_prev_prev_body_pose: Optional[np.ndarray] = None
    temporal_init_blend: float = 0.7
    temporal_extrapolation: float = 1.0
    w_temporal: float = 3e-3
    w_temporal_velocity: float = 2e-3
    w_temporal_accel: float = 5e-4
    bad_loss_threshold: float = 3e-5
    bad_data_loss_threshold: float = 2e-5
    bad_loss_growth_ratio: float = 1.5
    loss_divergence_ratio: float = 3.0
    min_valid_points: int = 6
    zero_weight_strategy: str = "uniform_finite"
    freeze_lower_body: bool = False
    init_prev_sim_scale: Optional[float] = None
    init_prev_sim_R: Optional[np.ndarray] = None
    init_prev_sim_t: Optional[np.ndarray] = None
    reuse_prev_similarity_when_freeze_lower_body: bool = True


@dataclass
class OptimizationRunResult:
    """Outputs from stage-3 optimization."""

    out_npy: Path
    debug_dir: Path
    best_cam: str
    loss_history: List[float]
    best_loss: float
    final_loss: float
    best_data_loss: float
    final_data_loss: float
    best_iter: int
    used_temporal_init: bool
    is_bad_loss: bool
    best_pose: np.ndarray
    sim_scale: float
    sim_R: np.ndarray
    sim_t: np.ndarray


@dataclass
class OptimizationRuntime:
    """Reusable runtime state for repeated optimizations on the same model."""

    device: torch.device
    head: Any
    hand_mask: torch.Tensor
    keep_mask: torch.Tensor


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


def load_body_pose_from_npy(npy_path: Optional[Path]) -> Optional[np.ndarray]:
    if npy_path is None:
        return None
    data = safe_numpy_item_load(npy_path.expanduser().resolve())
    if "body_pose_params" not in data:
        raise KeyError(f"Missing 'body_pose_params' in temporal init file: {npy_path}")
    return np.asarray(data["body_pose_params"], dtype=np.float32).reshape(-1)


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


def resolve_lower_body_pose_indices(pose_dim: int) -> np.ndarray:
    idxs = LOWER_BODY_POSE_IDXS_BY_DIM.get(int(pose_dim))
    if idxs is None:
        raise RuntimeError(
            f"freeze_lower_body enabled but no hardcoded lower-body index table for pose_dim={pose_dim}"
        )
    idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)
    idxs = idxs[(idxs >= 0) & (idxs < int(pose_dim))]
    return np.unique(idxs)


def _build_base_keep_mask(
    pose_dim: int,
    hand_mask_133: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(int(pose_dim), device=device, dtype=torch.float32)
    pose_dim = int(pose_dim)
    hand_mask_133 = hand_mask_133.to(device=device, dtype=torch.bool)

    if pose_dim == int(hand_mask_133.numel()):
        mask[hand_mask_133] = 0.0
        if pose_dim >= 3:
            mask[-3:] = 0.0
        return mask

    if pose_dim == 204:
        mapped = 6 + MHR_PARAM_HAND_IDXS_133[MHR_PARAM_HAND_IDXS_133 < 130]
        mapped = mapped[(mapped >= 0) & (mapped < pose_dim)]
        if mapped.size > 0:
            mapped_t = torch.from_numpy(mapped).to(device=device, dtype=torch.long)
            mask[mapped_t] = 0.0
        return mask

    copy_len = min(pose_dim, int(hand_mask_133.numel()))
    if copy_len > 0:
        hand_idx = torch.nonzero(hand_mask_133[:copy_len], as_tuple=False).flatten()
        if hand_idx.numel() > 0:
            mask[hand_idx] = 0.0
    if pose_dim >= 3:
        mask[-3:] = 0.0
    return mask


def _sanitize_subset_and_weights(
    gtM: np.ndarray,
    wM: np.ndarray,
    min_valid_points: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.asarray(gtM, dtype=np.float32)
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"Expected gtM shape (M,3), got {gt.shape}")
    w = np.asarray(wM, dtype=np.float32).reshape(-1)
    if w.shape[0] != gt.shape[0]:
        raise ValueError(
            f"Weight count does not match gt points: {w.shape[0]} vs {gt.shape[0]}"
        )

    finite = np.isfinite(gt).all(axis=1)
    w[~np.isfinite(w)] = 0.0
    w[~finite] = 0.0
    w = np.clip(w, 0.0, 1.0)

    min_pts = max(3, int(min_valid_points))
    nonzero = (w > 1e-8) & finite
    if int(nonzero.sum()) < min_pts:
        if strategy == "uniform_finite":
            w = np.where(finite, 1.0, 0.0).astype(np.float32)
            nonzero = finite.copy()
        elif strategy == "fail":
            raise RuntimeError(
                f"Insufficient valid weighted points: {int(nonzero.sum())} < {min_pts}"
            )
        else:
            raise ValueError(
                f"Unsupported zero_weight_strategy='{strategy}'. "
                "Expected one of: uniform_finite, fail."
            )

    if int(nonzero.sum()) < min_pts:
        raise RuntimeError(
            f"Insufficient finite points after fallback: {int(nonzero.sum())} < {min_pts}"
        )
    return finite, w, nonzero


def _resolve_valid_indices_for_prediction(
    predM: torch.Tensor,
    finite_gt_mask_t: torch.Tensor,
    base_wM_t: torch.Tensor,
    min_valid_points: int,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_finite_t = torch.isfinite(predM).all(dim=1)
    finite_pred_gt_t = finite_gt_mask_t & pred_finite_t
    w_masked = torch.where(finite_pred_gt_t, base_wM_t, torch.zeros_like(base_wM_t))
    valid_t = finite_pred_gt_t & (w_masked > 1e-8)
    min_pts = max(3, int(min_valid_points))

    if int(valid_t.sum().item()) < min_pts:
        if strategy == "uniform_finite":
            w_masked = torch.where(
                finite_pred_gt_t,
                torch.ones_like(base_wM_t),
                torch.zeros_like(base_wM_t),
            )
            valid_t = finite_pred_gt_t
        elif strategy == "fail":
            raise RuntimeError(
                f"Insufficient valid weighted points after pred finite mask: "
                f"{int(valid_t.sum().item())} < {min_pts}"
            )
        else:
            raise ValueError(
                f"Unsupported zero_weight_strategy='{strategy}'. "
                "Expected one of: uniform_finite, fail."
            )

    if int(valid_t.sum().item()) < min_pts:
        raise RuntimeError(
            "Insufficient valid points after applying prediction finite mask and "
            f"fallback: {int(valid_t.sum().item())} < {min_pts}"
        )

    idx_t = torch.nonzero(valid_t, as_tuple=False).flatten()
    return idx_t, w_masked


def safe_growth(final_value: float, best_value: float) -> float:
    if not np.isfinite(final_value):
        return float("inf")
    return float(final_value / max(best_value, 1e-12))


def classify_bad_optimization(
    config: OptimizationConfig,
    best_loss: float,
    final_loss: float,
    best_data_loss: float,
    final_data_loss: float,
) -> bool:
    total_growth = safe_growth(final_loss, best_loss)
    data_growth = safe_growth(final_data_loss, best_data_loss)
    return bool(
        (best_loss > float(config.bad_loss_threshold))
        or (best_data_loss > float(config.bad_data_loss_threshold))
        or (total_growth > float(config.bad_loss_growth_ratio))
        or (data_growth > float(config.bad_loss_growth_ratio))
    )


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


def build_optimization_runtime(config: OptimizationConfig) -> OptimizationRuntime:
    """Build reusable model/runtime tensors for repeated frame optimizations."""

    device = torch.device(config.device)
    head = load_sam_head(config, config.device)

    from sam_3d_body.models.modules.mhr_utils import mhr_param_hand_mask

    hand_mask = mhr_param_hand_mask.to(device)  # bool (133,)
    keep_mask = torch.ones(133, device=device, dtype=torch.float32)
    keep_mask[hand_mask] = 0.0
    keep_mask[-3:] = 0.0  # jaw

    return OptimizationRuntime(
        device=device,
        head=head,
        hand_mask=hand_mask,
        keep_mask=keep_mask,
    )


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
    """Build CLI parser for standalone stage-3 optimization."""

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
    ap.add_argument("--w_temporal", type=float, default=3e-3, help="Temporal pose prior weight when --init_pose_npy is provided.")
    ap.add_argument("--w_temporal_velocity", type=float, default=2e-3, help="Weight for temporal velocity target (pose extrapolation from previous 2 frames).")
    ap.add_argument("--w_temporal_accel", type=float, default=5e-4, help="Weight for temporal acceleration smoothing when previous 2 frames are available.")
    ap.add_argument("--temporal_init_blend", type=float, default=0.7, help="Blend between per-view init and temporal init pose.")
    ap.add_argument("--temporal_extrapolation", type=float, default=1.0, help="Extrapolation gain for temporal velocity target using prev and prev-prev poses.")
    ap.add_argument("--topk_print", type=int, default=10)
    ap.add_argument("--no_debug_artifacts", action="store_true", help="Skip debug plots/npz/ply for faster execution.")
    ap.add_argument("--min_iters", type=int, default=50, help="Minimum iterations before early stopping is allowed.")
    ap.add_argument("--early_stop_patience", type=int, default=60, help="Stop after this many non-improving iterations.")
    ap.add_argument("--early_stop_tol", type=float, default=1e-6, help="Minimum loss improvement to reset patience.")
    ap.add_argument("--init_pose_npy", type=Path, default=None, help="Optional .npy with body_pose_params used as temporal initialization.")
    ap.add_argument("--bad_loss_threshold", type=float, default=3e-5, help="Mark result bad when best total loss exceeds this value.")
    ap.add_argument("--bad_data_loss_threshold", type=float, default=2e-5, help="Mark result bad when best data loss exceeds this value.")
    ap.add_argument("--bad_loss_growth_ratio", type=float, default=1.5, help="Mark result bad when final_loss / best_loss exceeds this ratio.")
    ap.add_argument("--loss_divergence_ratio", type=float, default=3.0, help="Early break when current loss diverges above best_loss by this ratio.")
    ap.add_argument("--min_valid_points", type=int, default=6, help="Minimum valid points required for Umeyama/loss computations.")
    ap.add_argument(
        "--zero_weight_strategy",
        type=str,
        choices=["uniform_finite", "fail"],
        default="uniform_finite",
        help="Behavior when inlier weights collapse to ~zero on valid points.",
    )
    ap.add_argument(
        "--freeze_lower_body",
        action="store_true",
        help="Freeze lower-body pose dimensions to initialization (reduces stationary-leg jitter).",
    )
    return ap


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args for stage-3 optimization."""

    parser = build_arg_parser()
    return parser.parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> OptimizationConfig:
    """Convert parsed CLI args to `OptimizationConfig`."""

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
        w_temporal=args.w_temporal,
        w_temporal_velocity=args.w_temporal_velocity,
        w_temporal_accel=args.w_temporal_accel,
        temporal_init_blend=args.temporal_init_blend,
        temporal_extrapolation=args.temporal_extrapolation,
        topk_print=args.topk_print,
        save_debug_artifacts=not args.no_debug_artifacts,
        min_iters=args.min_iters,
        early_stop_patience=args.early_stop_patience,
        early_stop_tol=args.early_stop_tol,
        init_body_pose=load_body_pose_from_npy(args.init_pose_npy),
        bad_loss_threshold=args.bad_loss_threshold,
        bad_data_loss_threshold=args.bad_data_loss_threshold,
        bad_loss_growth_ratio=args.bad_loss_growth_ratio,
        loss_divergence_ratio=args.loss_divergence_ratio,
        min_valid_points=args.min_valid_points,
        zero_weight_strategy=args.zero_weight_strategy,
        freeze_lower_body=args.freeze_lower_body,
    )


def run_optimization(
    config: OptimizationConfig,
    runtime: Optional[OptimizationRuntime] = None,
) -> OptimizationRunResult:
    """Run optimization of body pose parameters using refined multi-view 3D points.

    Core idea:
    1. Score each camera-specific SAM initialization in 3D against triangulated GT.
    2. Pick the best initialization and optimize a shared body pose vector.
    3. Re-run MHR forward pass and write optimized mesh/keypoint outputs.
    """

    if not config.hf_repo and not config.ckpt:
        raise ValueError("Either hf_repo or ckpt must be provided.")

    if runtime is None:
        runtime = build_optimization_runtime(config)

    device = runtime.device
    head = runtime.head
    hand_mask = runtime.hand_mask
    runtime_keep_mask = runtime.keep_mask

    debug_dir = config.debug_dir.expanduser().resolve()
    if config.save_debug_artifacts:
        ensure_dir(debug_dir)
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

    finite_gt_mask_np, wM_np, _ = _sanitize_subset_and_weights(
        gtM=gtM,
        wM=wM,
        min_valid_points=config.min_valid_points,
        strategy=config.zero_weight_strategy,
    )
    finite_gt_mask_t = torch.from_numpy(finite_gt_mask_np).to(device=device, dtype=torch.bool)
    gtM_t = to_torch(gtM, device)
    wM_t = to_torch(wM_np, device)

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

        view_keep_mask = _build_base_keep_mask(
            pose_dim=int(pose133.numel()),
            hand_mask_133=hand_mask,
            device=device,
        )
        pose_eff = pose133 * view_keep_mask

        out = mhr_fk(head, pose_eff, hand108, scale28, shape45, expr72, device,
                     want_verts=False, want_joint=False, want_model_params=False)
        keypoints_308 = out[1].squeeze(0)   # (308,3)
        k70 = apply_repo_camera_flip_xyz(keypoints_308[:70])

        predM = k70[subset_idx]  # (M,3)
        try:
            valid_idx_t, wM_view_t = _resolve_valid_indices_for_prediction(
                predM=predM,
                finite_gt_mask_t=finite_gt_mask_t,
                base_wM_t=wM_t,
                min_valid_points=config.min_valid_points,
                strategy=config.zero_weight_strategy,
            )
        except RuntimeError:
            view_scores_3d[cam] = float("inf")
            view_align[cam] = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            print(f"[score] {cam:>8s}: 3D=inf m   mean_px={view_mean_px[cam]:.3f}   file={npy_path.name}")
            continue

        predM_v = predM.index_select(0, valid_idx_t)
        gtM_v = gtM_t.index_select(0, valid_idx_t)
        wM_v = wM_view_t.index_select(0, valid_idx_t)

        s, R, t = umeyama_similarity(predM_v, gtM_v, w=wM_v, with_scale=config.with_scale)
        predM_aligned_v = s * (predM_v @ R.T) + t[None, :]
        r = torch.sqrt(((predM_aligned_v - gtM_v) ** 2).sum(dim=1) + 1e-12)
        score = (wM_v * r).sum() / (wM_v.sum() + 1e-9)

        view_scores_3d[cam] = float(score.detach().cpu().item())
        view_align[cam] = (float(s.cpu().item()), R.cpu().numpy(), t.cpu().numpy())

        print(f"[score] {cam:>8s}: 3D={view_scores_3d[cam]:.6f} m   mean_px={view_mean_px[cam]:.3f}   file={npy_path.name}")

    best_cam = sorted(cams, key=lambda c: (view_scores_3d[c], view_mean_px[c]))[0]
    print(f"\n[init] best_cam = {best_cam}  (3D={view_scores_3d[best_cam]:.6f} m, mean_px={view_mean_px[best_cam]:.3f})")

    init_dict = view_dicts[best_cam]

    init_pose_raw = to_torch(init_dict["body_pose_params"], device).flatten().to(torch.float32)
    init_hand = get_param_array(init_dict, "hand_pose_params", device)
    init_scale = get_param_array(init_dict, "scale_params", device)
    init_shape = get_param_array(init_dict, "shape_params", device)
    init_expr = get_param_array(init_dict, "expr_params", device)

    pose_dim = int(init_pose_raw.numel())
    if int(runtime_keep_mask.numel()) == pose_dim:
        base_keep_mask = runtime_keep_mask.to(device=device, dtype=torch.float32)
    else:
        base_keep_mask = _build_base_keep_mask(
            pose_dim=pose_dim,
            hand_mask_133=hand_mask,
            device=device,
        )

    optimize_mask = base_keep_mask.clone()
    if config.freeze_lower_body:
        lower_idxs_np = resolve_lower_body_pose_indices(pose_dim=pose_dim)
        if lower_idxs_np.size > 0:
            lower_idxs_t = torch.from_numpy(lower_idxs_np).to(device=device, dtype=torch.long)
            optimize_mask[lower_idxs_t] = 0.0

    init_pose_ref = init_pose_raw.clone()
    frozen_pose_target = init_pose_raw.clone()
    temporal_prev_np = config.init_prev_body_pose
    if temporal_prev_np is None:
        # Backward-compatibility for existing callers.
        temporal_prev_np = config.init_body_pose
    temporal_prev_prev_np = config.init_prev_prev_body_pose

    temporal_pose = None
    temporal_prev_prev_pose = None
    temporal_velocity_target = None
    used_temporal_init = temporal_prev_np is not None
    if temporal_prev_np is not None:
        temporal_pose = to_torch(temporal_prev_np, device).flatten().to(torch.float32)
        if int(temporal_pose.numel()) != pose_dim:
            raise ValueError(
                f"Temporal pose dim mismatch: expected {pose_dim}, got {int(temporal_pose.numel())}"
            )
        init_target = temporal_pose
        if temporal_prev_prev_np is not None:
            temporal_prev_prev_pose = to_torch(temporal_prev_prev_np, device).flatten().to(torch.float32)
            if int(temporal_prev_prev_pose.numel()) != pose_dim:
                raise ValueError(
                    "Temporal prev-prev pose dim mismatch: "
                    f"expected {pose_dim}, got {int(temporal_prev_prev_pose.numel())}"
                )
            extrap = float(np.clip(config.temporal_extrapolation, 0.0, 2.0))
            temporal_velocity_target = temporal_pose + extrap * (temporal_pose - temporal_prev_prev_pose)
            init_target = temporal_velocity_target
        blend = float(np.clip(config.temporal_init_blend, 0.0, 1.0))
        blend_mask = optimize_mask * blend
        init_pose_ref = init_pose_ref * (1.0 - blend_mask) + init_target * blend_mask
        if config.freeze_lower_body:
            frozen_pose_target = torch.where(
                optimize_mask == 0,
                temporal_pose,
                frozen_pose_target,
            )

    # ---- Optimize pose (leaf tensor) ----
    pose = init_pose_ref.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pose], lr=config.lr)

    loss_hist: List[float] = []
    data_loss_hist: List[float] = []

    # debug init plot
    if config.save_debug_artifacts:
        with torch.no_grad():
            pose_eff0 = pose.detach() * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)
            out0 = mhr_fk(head, pose_eff0, init_hand, init_scale, init_shape, init_expr, device,
                          want_verts=False, want_joint=False, want_model_params=False)
            k70_0 = apply_repo_camera_flip_xyz(out0[1].squeeze(0)[:70])
            predM0 = k70_0[subset_idx]
            valid_idx0_t, wM0_t = _resolve_valid_indices_for_prediction(
                predM=predM0,
                finite_gt_mask_t=finite_gt_mask_t,
                base_wM_t=wM_t,
                min_valid_points=config.min_valid_points,
                strategy=config.zero_weight_strategy,
            )
            predM0_v = predM0.index_select(0, valid_idx0_t)
            gtM0_v = gtM_t.index_select(0, valid_idx0_t)
            wM0_v = wM0_t.index_select(0, valid_idx0_t)
            s0, R0, t0 = umeyama_similarity(predM0_v, gtM0_v, w=wM0_v, with_scale=config.with_scale)
            predM0_al = s0 * (predM0_v @ R0.T) + t0[None, :]
            plot_3d_compare(
                gtM[valid_idx0_t.cpu().numpy()],
                predM0_al.cpu().numpy(),
                f"Init ({best_cam}) aligned to GT (M={M})",
                debug_dir / "compare_init_subset.png",
            )

    # main loop
    min_iters = max(1, min(int(config.min_iters), int(config.iters)))
    patience = max(1, int(config.early_stop_patience))
    improve_tol = max(0.0, float(config.early_stop_tol))
    divergence_ratio = max(1.0, float(config.loss_divergence_ratio))
    best_loss = float("inf")
    best_data_loss = float("inf")
    best_iter = -1
    best_pose = pose.detach().clone()
    no_improve = 0
    for it in range(config.iters):
        opt.zero_grad(set_to_none=True)

        pose_eff = pose * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)

        out = mhr_fk(head, pose_eff, init_hand, init_scale, init_shape, init_expr, device,
                     want_verts=False, want_joint=False, want_model_params=False)
        k70 = apply_repo_camera_flip_xyz(out[1].squeeze(0)[:70])
        predM = k70[subset_idx]

        valid_idx_t, wM_masked_t = _resolve_valid_indices_for_prediction(
            predM=predM,
            finite_gt_mask_t=finite_gt_mask_t,
            base_wM_t=wM_t,
            min_valid_points=config.min_valid_points,
            strategy=config.zero_weight_strategy,
        )

        predM_v = predM.index_select(0, valid_idx_t)
        gtM_v = gtM_t.index_select(0, valid_idx_t)
        wM_v = wM_masked_t.index_select(0, valid_idx_t)

        with torch.no_grad():
            s, R, t = umeyama_similarity(predM_v.detach(), gtM_v, w=wM_v, with_scale=config.with_scale)

        predM_aligned = s * (predM_v @ R.T) + t[None, :]
        diff = predM_aligned - gtM_v
        r = torch.sqrt((diff * diff).sum(dim=1) + 1e-12)

        loss_data = (wM_v * huber(r, delta=config.huber_m)).sum() / (wM_v.sum() + 1e-9)
        loss_reg = config.w_pose_reg * torch.mean((pose - init_pose_ref) ** 2)
        if temporal_pose is not None and config.w_temporal > 0:
            loss_temporal = config.w_temporal * torch.mean((pose - temporal_pose) ** 2)
        else:
            loss_temporal = torch.zeros((), device=device, dtype=torch.float32)
        if temporal_velocity_target is not None and config.w_temporal_velocity > 0:
            loss_temporal_velocity = config.w_temporal_velocity * torch.mean((pose - temporal_velocity_target) ** 2)
        else:
            loss_temporal_velocity = torch.zeros((), device=device, dtype=torch.float32)
        if (
            temporal_pose is not None
            and temporal_prev_prev_pose is not None
            and config.w_temporal_accel > 0
        ):
            prev_vel = temporal_pose - temporal_prev_prev_pose
            cur_vel = pose - temporal_pose
            loss_temporal_accel = config.w_temporal_accel * torch.mean((cur_vel - prev_vel) ** 2)
        else:
            loss_temporal_accel = torch.zeros((), device=device, dtype=torch.float32)
        loss = loss_data + loss_reg + loss_temporal + loss_temporal_velocity + loss_temporal_accel

        loss.backward()

        with torch.no_grad():
            if pose.grad is not None:
                pose.grad[optimize_mask == 0] = 0.0

        torch.nn.utils.clip_grad_norm_([pose], 1.0)
        opt.step()

        with torch.no_grad():
            pose.copy_(pose * optimize_mask + frozen_pose_target * (1.0 - optimize_mask))

        loss_hist.append(float(loss.detach().cpu().item()))
        data_loss_hist.append(float(loss_data.detach().cpu().item()))
        if it % 25 == 0 or it == config.iters - 1:
            print(
                f"[{it:04d}] loss={loss_hist[-1]:.6f} "
                f"data={float(loss_data.detach().cpu().item()):.6f} "
                f"temporal={float(loss_temporal.detach().cpu().item()):.6f} "
                f"vel={float(loss_temporal_velocity.detach().cpu().item()):.6f} "
                f"accel={float(loss_temporal_accel.detach().cpu().item()):.6f}"
            )

        cur_loss = loss_hist[-1]
        cur_data_loss = data_loss_hist[-1]
        if cur_loss < (best_loss - improve_tol):
            best_loss = cur_loss
            best_iter = it
            best_pose = pose.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
        if cur_data_loss < best_data_loss:
            best_data_loss = cur_data_loss
        if (it + 1) >= min_iters and np.isfinite(best_loss) and cur_loss > (best_loss * divergence_ratio):
            print(
                f"[diverge-stop] iter={it:04d} cur_loss={cur_loss:.6f} "
                f"best_loss={best_loss:.6f} ratio={cur_loss / max(best_loss, 1e-12):.2f}"
            )
            break
        if (it + 1) >= min_iters and no_improve >= patience:
            print(f"[early-stop] iter={it:04d} best_loss={best_loss:.6f}")
            break

    final_loss = float(loss_hist[-1]) if loss_hist else float("inf")
    final_data_loss = float(data_loss_hist[-1]) if data_loss_hist else float("inf")
    if best_iter >= 0:
        with torch.no_grad():
            pose.copy_(best_pose)
            pose.copy_(pose * optimize_mask + frozen_pose_target * (1.0 - optimize_mask))

    if config.save_debug_artifacts:
        plot_loss_curve(loss_hist, debug_dir / "loss_curve.png")

    output_sim_scale = 1.0
    output_sim_R = np.eye(3, dtype=np.float32)
    output_sim_t = np.zeros(3, dtype=np.float32)

    # final forward (full outputs)
    with torch.no_grad():
        pose_eff = pose.detach() * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)
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

        valid_idx_t, wM_masked_t = _resolve_valid_indices_for_prediction(
            predM=predM,
            finite_gt_mask_t=finite_gt_mask_t,
            base_wM_t=wM_t,
            min_valid_points=config.min_valid_points,
            strategy=config.zero_weight_strategy,
        )
        predM_v = predM.index_select(0, valid_idx_t)
        gtM_v = gtM_t.index_select(0, valid_idx_t)
        wM_v = wM_masked_t.index_select(0, valid_idx_t)

        fit_s, fit_R, fit_t = umeyama_similarity(predM_v, gtM_v, w=wM_v, with_scale=config.with_scale)
        out_s, out_R, out_t = fit_s, fit_R, fit_t
        reused_prev_similarity = False
        if (
            config.freeze_lower_body
            and config.reuse_prev_similarity_when_freeze_lower_body
            and config.init_prev_sim_scale is not None
            and config.init_prev_sim_R is not None
            and config.init_prev_sim_t is not None
        ):
            prev_R = np.asarray(config.init_prev_sim_R, dtype=np.float32).reshape(3, 3)
            prev_t = np.asarray(config.init_prev_sim_t, dtype=np.float32).reshape(3)
            prev_s = float(config.init_prev_sim_scale)
            if np.isfinite(prev_s) and np.isfinite(prev_R).all() and np.isfinite(prev_t).all():
                out_s = to_torch(prev_s, device)
                out_R = to_torch(prev_R, device)
                out_t = to_torch(prev_t, device)
                reused_prev_similarity = True

        k70_aligned = out_s * (k70 @ out_R.T) + out_t[None, :]
        verts_aligned = out_s * (verts @ out_R.T) + out_t[None, :]
        jcoords_aligned = out_s * (jcoords @ out_R.T) + out_t[None, :]
        jrots_aligned = out_R[None, :, :] @ jrots  # optional consistency

        predM_aligned_v = fit_s * (predM_v @ fit_R.T) + fit_t[None, :]
        residual_subset = torch.sqrt(((predM_aligned_v - gtM_v) ** 2).sum(dim=1) + 1e-12)
        aligned_data_loss = (wM_v * huber(residual_subset, delta=config.huber_m)).sum() / (wM_v.sum() + 1e-9)
        final_data_loss = float(aligned_data_loss.detach().cpu().item())
        best_data_loss = min(best_data_loss, final_data_loss)
        final_reg_loss = float((config.w_pose_reg * torch.mean((pose - init_pose_ref) ** 2)).detach().cpu().item())
        if temporal_pose is not None and config.w_temporal > 0:
            final_temporal = float(
                (config.w_temporal * torch.mean((pose - temporal_pose) ** 2)).detach().cpu().item()
            )
        else:
            final_temporal = 0.0
        if temporal_velocity_target is not None and config.w_temporal_velocity > 0:
            final_temporal_velocity = float(
                (config.w_temporal_velocity * torch.mean((pose - temporal_velocity_target) ** 2))
                .detach()
                .cpu()
                .item()
            )
        else:
            final_temporal_velocity = 0.0
        if (
            temporal_pose is not None
            and temporal_prev_prev_pose is not None
            and config.w_temporal_accel > 0
        ):
            prev_vel = temporal_pose - temporal_prev_prev_pose
            cur_vel = pose - temporal_pose
            final_temporal_accel = float(
                (config.w_temporal_accel * torch.mean((cur_vel - prev_vel) ** 2)).detach().cpu().item()
            )
        else:
            final_temporal_accel = 0.0
        final_loss = (
            final_data_loss
            + final_reg_loss
            + final_temporal
            + final_temporal_velocity
            + final_temporal_accel
        )

        rM = residual_subset.cpu().numpy()
        worst = np.argsort(-rM)[: config.topk_print]
        print("\nTop residual points (subset indices):")
        for idx in worst:
            subset_pos = int(valid_idx_t[idx].item())
            name = str(subset_names[subset_pos]) if subset_names is not None else f"pt{subset_pos}"
            print(f"  {idx:02d} ({name}): {rM[idx]:.4f} m")

        if config.save_debug_artifacts:
            plot_3d_compare(
                gtM[valid_idx_t.cpu().numpy()],
                predM_aligned_v.cpu().numpy(),
                f"Optimized aligned to GT (M={int(valid_idx_t.numel())})",
                debug_dir / "compare_opt_subset.png",
            )

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
                w_subset=wM_np,
                init_scores_3d_m=np.array([view_scores_3d[c] for c in cams], dtype=np.float32),
                init_scores_mean_px=np.array([view_mean_px[c] for c in cams], dtype=np.float32),
                final_scale=np.array(float(out_s.cpu().item()), dtype=np.float32),
                final_R=out_R.cpu().numpy(),
                final_t=out_t.cpu().numpy(),
                final_scale_fit=np.array(float(fit_s.cpu().item()), dtype=np.float32),
                final_R_fit=fit_R.cpu().numpy(),
                final_t_fit=fit_t.cpu().numpy(),
                reused_prev_similarity=np.array(int(reused_prev_similarity), dtype=np.int32),
                loss_hist=np.array(loss_hist, dtype=np.float32),
                data_loss_hist=np.array(data_loss_hist, dtype=np.float32),
                best_loss=np.array(best_loss, dtype=np.float32),
                final_loss=np.array(final_loss, dtype=np.float32),
                best_data_loss=np.array(best_data_loss, dtype=np.float32),
                final_data_loss=np.array(final_data_loss, dtype=np.float32),
                best_iter=np.array(best_iter, dtype=np.int32),
                residual_subset_m=np.array(rM, dtype=np.float32),
                keep_mask=optimize_mask.cpu().numpy(),
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
        out_dict["opt_sim_scale"] = float(out_s.cpu().item())
        out_dict["opt_sim_R"] = out_R.cpu().numpy()
        out_dict["opt_sim_t"] = out_t.cpu().numpy()
        out_dict["opt_sim_scale_fit"] = float(fit_s.cpu().item())
        out_dict["opt_sim_R_fit"] = fit_R.cpu().numpy()
        out_dict["opt_sim_t_fit"] = fit_t.cpu().numpy()
        out_dict["opt_sim_reused_prev"] = int(reused_prev_similarity)
        out_dict["opt_loss_hist"] = np.array(loss_hist, dtype=np.float32)
        out_dict["opt_data_loss_hist"] = np.array(data_loss_hist, dtype=np.float32)
        out_dict["opt_best_loss"] = float(best_loss)
        out_dict["opt_final_loss"] = float(final_loss)
        out_dict["opt_best_data_loss"] = float(best_data_loss)
        out_dict["opt_final_data_loss"] = float(final_data_loss)
        out_dict["opt_best_iter"] = int(best_iter)
        growth = safe_growth(final_loss, best_loss)
        data_growth = safe_growth(final_data_loss, best_data_loss)
        out_dict["opt_loss_growth_ratio"] = growth
        out_dict["opt_data_loss_growth_ratio"] = data_growth
        out_dict["opt_used_temporal_init"] = int(used_temporal_init)
        out_dict["opt_temporal_weight"] = float(config.w_temporal)
        out_dict["opt_temporal_velocity_weight"] = float(config.w_temporal_velocity)
        out_dict["opt_temporal_accel_weight"] = float(config.w_temporal_accel)
        out_dict["opt_temporal_extrapolation"] = float(config.temporal_extrapolation)
        out_dict["opt_subset_indices"] = subset_idx
        out_dict["opt_points3d_refined"] = gtM

        is_bad_loss = classify_bad_optimization(
            config=config,
            best_loss=best_loss,
            final_loss=final_loss,
            best_data_loss=best_data_loss,
            final_data_loss=final_data_loss,
        )
        out_dict["opt_is_bad_loss"] = int(is_bad_loss)

        np.save(out_npy, out_dict, allow_pickle=True)
        output_sim_scale = float(out_s.cpu().item())
        output_sim_R = out_R.cpu().numpy().astype(np.float32)
        output_sim_t = out_t.cpu().numpy().astype(np.float32)

    print(f"\nSaved optimized npy: {out_npy}")
    print(
        f"[quality] best_loss={best_loss:.6f} final_loss={final_loss:.6f} "
        f"best_data={best_data_loss:.6f} final_data={final_data_loss:.6f} "
        f"best_iter={best_iter} temporal_init={int(used_temporal_init)}"
    )
    if config.save_debug_artifacts:
        print(f"Saved debug dir: {debug_dir}")
    else:
        print("Debug artifacts disabled (--no_debug_artifacts).")
    is_bad_loss = classify_bad_optimization(
        config=config,
        best_loss=best_loss,
        final_loss=final_loss,
        best_data_loss=best_data_loss,
        final_data_loss=final_data_loss,
    )
    return OptimizationRunResult(
        out_npy=out_npy,
        debug_dir=debug_dir,
        best_cam=best_cam,
        loss_history=loss_hist,
        best_loss=float(best_loss),
        final_loss=float(final_loss),
        best_data_loss=float(best_data_loss),
        final_data_loss=float(final_data_loss),
        best_iter=int(best_iter),
        used_temporal_init=bool(used_temporal_init),
        is_bad_loss=is_bad_loss,
        best_pose=(pose.detach() * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)).cpu().numpy().astype(np.float32),
        sim_scale=output_sim_scale,
        sim_R=output_sim_R,
        sim_t=output_sim_t,
    )


def main(args: Optional[argparse.Namespace] = None) -> OptimizationRunResult:
    """CLI/programmatic entrypoint for stage-3 optimization."""

    if args is None:
        args = parse_args()
    config = namespace_to_config(args)
    return run_optimization(config)


if __name__ == "__main__":
    main()
