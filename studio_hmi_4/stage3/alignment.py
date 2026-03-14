"""Alignment, masking, and loss-quality helpers for stage-3 optimization."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from .types import OptimizationConfig


MHR_PARAM_HAND_IDXS_133 = np.array(
    [
        62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115,
    ],
    dtype=np.int64,
)

LOWER_BODY_POSE_IDXS_BY_DIM = {
    # Conservative lower-body-only subset verified by perturbation using
    # inspect_body_pose_param_effects.py at delta=0.05 and delta=0.5.
    # We intentionally keep only indices that consistently move lower-body
    # keypoints, instead of the earlier broad heuristic range.
    133: np.array(
        [
            44, 45, 46, 47, 48, 49,
            53, 54, 55, 56, 57, 58,
            116, 117, 118, 120, 121, 122,
            128, 129,
        ],
        dtype=np.int64,
    ),
    204: np.array(
        sorted(
            set(
                list(range(0, 6))
                + [
                    6 + i
                    for i in [
                        44, 45, 46, 47, 48, 49,
                        53, 54, 55, 56, 57, 58,
                        116, 117, 118, 120, 121, 122,
                        128, 129,
                    ]
                ]
            )
        ),
        dtype=np.int64,
    ),
}

ALIGNMENT_ANCHOR_NAMES = {
    "left_hip",
    "right_hip",
    "neck",
    "left_acromion",
    "right_acromion",
}

_FACE_KEYPOINT_NAMES = {
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
}

_TORSO_KEYPOINT_NAMES = {
    "neck",
    "left_hip",
    "right_hip",
}

_LOWER_BODY_KEYPOINT_NAMES = {
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
}

_ARM_ANCHOR_KEYPOINT_NAMES = {
    "left_shoulder",
    "right_shoulder",
    "left_acromion",
    "right_acromion",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
}

_ARM_JOINT_KEYPOINT_NAMES = {
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
}

_HAND_KEYPOINT_NAMES = {
    "right_thumb4",
    "right_thumb3",
    "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4",
    "right_forefinger3",
    "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4",
    "right_middle_finger3",
    "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4",
    "right_ring_finger3",
    "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4",
    "right_pinky_finger3",
    "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "left_thumb4",
    "left_thumb3",
    "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4",
    "left_forefinger3",
    "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4",
    "left_middle_finger3",
    "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4",
    "left_ring_finger3",
    "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4",
    "left_pinky_finger3",
    "left_pinky_finger2",
    "left_pinky_finger_third_joint",
}


def _build_default_subset_loss_weight_by_name() -> dict[str, float]:
    weights: dict[str, float] = {}
    for name in _FACE_KEYPOINT_NAMES:
        weights[name] = 0.05
    for name in _TORSO_KEYPOINT_NAMES:
        weights[name] = 0.15
    for name in _LOWER_BODY_KEYPOINT_NAMES:
        weights[name] = 0.10
    for name in _ARM_ANCHOR_KEYPOINT_NAMES:
        weights[name] = 0.25
    for name in _ARM_JOINT_KEYPOINT_NAMES:
        weights[name] = 0.60
    for name in _HAND_KEYPOINT_NAMES:
        weights[name] = 1.00
    return weights


SUBSET_LOSS_WEIGHT_BY_NAME = _build_default_subset_loss_weight_by_name()


@torch.no_grad()
def umeyama_similarity(X, Y, w=None, with_scale=True, eps=1e-9):
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

    S = (Xc * w[:, None]).T @ Yc
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
    return torch.where(r <= delta, 0.5 * r * r, delta * (r - 0.5 * delta))


def resolve_lower_body_pose_indices(pose_dim: int) -> np.ndarray:
    idxs = LOWER_BODY_POSE_IDXS_BY_DIM.get(int(pose_dim))
    if idxs is None:
        raise RuntimeError(
            f"freeze_lower_body enabled but no hardcoded lower-body index table for pose_dim={pose_dim}"
        )
    idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)
    idxs = idxs[(idxs >= 0) & (idxs < int(pose_dim))]
    return np.unique(idxs)


def build_base_keep_mask(
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


def build_subset_loss_weights(
    subset_names: Optional[np.ndarray],
) -> np.ndarray:
    if subset_names is None:
        return np.ones((0,), dtype=np.float32)
    names = np.asarray(subset_names).reshape(-1)
    weights = np.ones((int(names.shape[0]),), dtype=np.float32)
    for i, name in enumerate(names):
        weights[i] = float(SUBSET_LOSS_WEIGHT_BY_NAME.get(str(name), 1.0))
    return weights


def sanitize_subset_and_weights(
    gtM: np.ndarray,
    wM: np.ndarray,
    min_valid_points: int,
    strategy: str,
    allowed_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.asarray(gtM, dtype=np.float32)
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"Expected gtM shape (M,3), got {gt.shape}")
    w = np.asarray(wM, dtype=np.float32).reshape(-1)
    if w.shape[0] != gt.shape[0]:
        raise ValueError(f"Weight count does not match gt points: {w.shape[0]} vs {gt.shape[0]}")

    finite = np.isfinite(gt).all(axis=1)
    allowed = np.ones_like(finite, dtype=bool)
    if allowed_mask is not None:
        allowed = np.asarray(allowed_mask, dtype=bool).reshape(-1)
        if allowed.shape[0] != gt.shape[0]:
            raise ValueError(f"Allowed-mask count does not match gt points: {allowed.shape[0]} vs {gt.shape[0]}")
    w[~np.isfinite(w)] = 0.0
    w[~finite] = 0.0
    w[~allowed] = 0.0
    w = np.clip(w, 0.0, 1.0)

    min_pts = max(3, int(min_valid_points))
    nonzero = (w > 1e-8) & finite & allowed
    if int(nonzero.sum()) < min_pts:
        if strategy == "uniform_finite":
            w = np.where(finite & allowed, 1.0, 0.0).astype(np.float32)
            nonzero = (finite & allowed).copy()
        elif strategy == "fail":
            raise RuntimeError(f"Insufficient valid weighted points: {int(nonzero.sum())} < {min_pts}")
        else:
            raise ValueError(
                f"Unsupported zero_weight_strategy='{strategy}'. Expected one of: uniform_finite, fail."
            )

    if int(nonzero.sum()) < min_pts:
        raise RuntimeError(f"Insufficient finite points after fallback: {int(nonzero.sum())} < {min_pts}")
    return finite, w, nonzero


def resolve_valid_indices_for_prediction(
    predM: torch.Tensor,
    finite_gt_mask_t: torch.Tensor,
    base_wM_t: torch.Tensor,
    min_valid_points: int,
    strategy: str,
    allowed_mask_t: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_finite_t = torch.isfinite(predM).all(dim=1)
    finite_pred_gt_t = finite_gt_mask_t & pred_finite_t
    if allowed_mask_t is not None:
        finite_pred_gt_t = finite_pred_gt_t & allowed_mask_t
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
                f"Insufficient valid weighted points after pred finite mask: {int(valid_t.sum().item())} < {min_pts}"
            )
        else:
            raise ValueError(
                f"Unsupported zero_weight_strategy='{strategy}'. Expected one of: uniform_finite, fail."
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


def build_alignment_anchor_local_indices(
    subset_names: Optional[np.ndarray],
) -> np.ndarray:
    if subset_names is None:
        return np.zeros((0,), dtype=np.int64)
    names = np.asarray(subset_names).reshape(-1)
    idxs: List[int] = []
    for i, name in enumerate(names):
        if str(name) in ALIGNMENT_ANCHOR_NAMES:
            idxs.append(int(i))
    if len(idxs) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(sorted(set(idxs)), dtype=np.int64)


def select_alignment_subset_tensors(
    predM_v: torch.Tensor,
    gtM_v: torch.Tensor,
    wM_v: torch.Tensor,
    valid_idx_t: torch.Tensor,
    anchor_local_idx_t: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if anchor_local_idx_t is None or int(anchor_local_idx_t.numel()) == 0:
        return predM_v, gtM_v, wM_v
    if int(valid_idx_t.numel()) == 0:
        return predM_v, gtM_v, wM_v
    anchor_mask = (valid_idx_t[:, None] == anchor_local_idx_t[None, :]).any(dim=1)
    if int(anchor_mask.sum().item()) >= 3:
        return predM_v[anchor_mask], gtM_v[anchor_mask], wM_v[anchor_mask]
    return predM_v, gtM_v, wM_v
