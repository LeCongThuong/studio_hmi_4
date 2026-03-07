"""Runtime/model-loading helpers for stage-3 optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from studio_hmi_4.common import load_npy_dict

from .types import OptimizationConfig, OptimizationRuntime


DEFAULT_PARAM_DIMS = {
    "hand_pose_params": 108,
    "scale_params": 28,
    "shape_params": 45,
    "expr_params": 72,
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_torch(x, device, dtype=torch.float32):
    return torch.tensor(x, device=device, dtype=dtype)


def safe_numpy_item_load(npy_path: Path):
    return load_npy_dict(npy_path)


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


def load_sam_head(config: OptimizationConfig, device_str: str):
    from sam_3d_body.build_models import load_sam_3d_body, load_sam_3d_body_hf

    if config.hf_repo:
        model, _cfg = load_sam_3d_body_hf(config.hf_repo, device=device_str)
    else:
        model, _cfg = load_sam_3d_body(
            checkpoint_path=config.ckpt,
            device=device_str,
            mhr_path=config.mhr_pt,
        )
    head = model.head_pose
    head.eval()
    return head


def build_optimization_runtime(config: OptimizationConfig) -> OptimizationRuntime:
    device = torch.device(config.device)
    head = load_sam_head(config, config.device)

    from sam_3d_body.models.modules.mhr_utils import mhr_param_hand_mask

    hand_mask = mhr_param_hand_mask.to(device)
    keep_mask = torch.ones(133, device=device, dtype=torch.float32)
    keep_mask[hand_mask] = 0.0
    keep_mask[-3:] = 0.0

    return OptimizationRuntime(
        device=device,
        head=head,
        hand_mask=hand_mask,
        keep_mask=keep_mask,
    )


def apply_repo_camera_flip_xyz(x):
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
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"mhr_forward failed for both pose sizes. Last error: {last_err}")
