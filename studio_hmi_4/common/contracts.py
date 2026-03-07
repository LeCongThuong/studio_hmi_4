"""Explicit shape contracts for stage-to-stage artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


class ContractError(ValueError):
    """Raised when a stage artifact does not match the expected schema."""


def _require_key(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ContractError(f"Missing required key '{key}'.")
    return data[key]


def _as_vector(
    data: Mapping[str, Any],
    key: str,
    expected_dim: int,
    required: bool,
) -> Optional[np.ndarray]:
    if key not in data:
        if required:
            raise ContractError(f"Missing required key '{key}'.")
        return None
    arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if arr.size != int(expected_dim):
        raise ContractError(f"Key '{key}' has dim {arr.size}, expected {expected_dim}.")
    if not np.isfinite(arr).all():
        raise ContractError(f"Key '{key}' contains non-finite values.")
    return arr


def _as_points2d(
    data: Mapping[str, Any],
    key: str,
    num_points: int,
    required: bool,
) -> Optional[np.ndarray]:
    if key not in data:
        if required:
            raise ContractError(f"Missing required key '{key}'.")
        return None
    arr = np.asarray(data[key], dtype=np.float32)
    is_valid = (
        (arr.ndim == 2 and arr.shape == (num_points, 2))
        or (arr.ndim == 3 and arr.shape[-2:] == (num_points, 2))
    )
    if not is_valid:
        raise ContractError(
            f"Key '{key}' must have shape ({num_points},2) or (N,{num_points},2), got {arr.shape}."
        )
    return arr


def _as_points3d(
    data: Mapping[str, Any],
    key: str,
    num_points: Optional[int],
    required: bool,
) -> Optional[np.ndarray]:
    if key not in data:
        if required:
            raise ContractError(f"Missing required key '{key}'.")
        return None
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ContractError(f"Key '{key}' must have shape (N,3), got {arr.shape}.")
    if num_points is not None and arr.shape[0] != int(num_points):
        raise ContractError(
            f"Key '{key}' must have {num_points} rows, got {arr.shape[0]}."
        )
    return arr


@dataclass(frozen=True)
class Stage1PredictionContract:
    pred_keypoints_2d: np.ndarray
    body_pose_params: Optional[np.ndarray]
    hand_pose_params: Optional[np.ndarray]
    scale_params: Optional[np.ndarray]
    shape_params: Optional[np.ndarray]
    expr_params: Optional[np.ndarray]


def validate_stage1_prediction_dict(
    data: Mapping[str, Any],
    require_pose_blocks: bool = False,
) -> Stage1PredictionContract:
    """Validate Stage-1 output shape-level contracts."""

    return Stage1PredictionContract(
        pred_keypoints_2d=_as_points2d(data, "pred_keypoints_2d", num_points=70, required=True),
        body_pose_params=_as_vector(data, "body_pose_params", expected_dim=133, required=require_pose_blocks),
        hand_pose_params=_as_vector(data, "hand_pose_params", expected_dim=108, required=False),
        scale_params=_as_vector(data, "scale_params", expected_dim=28, required=False),
        shape_params=_as_vector(data, "shape_params", expected_dim=45, required=False),
        expr_params=_as_vector(data, "expr_params", expected_dim=72, required=False),
    )


@dataclass(frozen=True)
class TriangulationBundleContract:
    points3d_refined: np.ndarray
    subset_indices: np.ndarray
    subset_names: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]


def validate_triangulation_bundle(data: Mapping[str, Any]) -> TriangulationBundleContract:
    """Validate Stage-2 bundle shapes."""

    subset_indices = np.asarray(_require_key(data, "subset_indices"), dtype=np.int64).reshape(-1)
    points3d_refined = _as_points3d(data, "points3d_refined", num_points=subset_indices.size, required=True)
    subset_names = None
    if "subset_names" in data:
        subset_names = np.asarray(data["subset_names"]).reshape(-1)
        if subset_names.shape[0] != subset_indices.shape[0]:
            raise ContractError(
                f"subset_names has {subset_names.shape[0]} entries, expected {subset_indices.shape[0]}."
            )
    inlier_mask = None
    if "inlier_mask" in data:
        inlier_mask = np.asarray(data["inlier_mask"], dtype=np.float32)
        if inlier_mask.ndim != 2 or inlier_mask.shape[0] != subset_indices.shape[0]:
            raise ContractError(
                "inlier_mask must have shape (M,V) with M matching subset_indices."
            )
    return TriangulationBundleContract(
        points3d_refined=points3d_refined,
        subset_indices=subset_indices,
        subset_names=subset_names,
        inlier_mask=inlier_mask,
    )


@dataclass(frozen=True)
class OptimizationResultContract:
    body_pose_params: np.ndarray
    hand_pose_params: Optional[np.ndarray]
    pred_keypoints_3d: np.ndarray
    pred_vertices: Optional[np.ndarray]
    mhr_model_params: Optional[np.ndarray]


def validate_optimization_result_dict(
    data: Mapping[str, Any],
    require_geometry: bool = False,
    require_mhr_compact: bool = False,
) -> OptimizationResultContract:
    """Validate Stage-3 output shape-level contracts."""

    pred_vertices = _as_points3d(data, "pred_vertices", num_points=None, required=require_geometry)
    pred_keypoints_3d = _as_points3d(data, "pred_keypoints_3d", num_points=70, required=True)
    return OptimizationResultContract(
        body_pose_params=_as_vector(data, "body_pose_params", expected_dim=133, required=True),
        hand_pose_params=_as_vector(data, "hand_pose_params", expected_dim=108, required=False),
        pred_keypoints_3d=pred_keypoints_3d,
        pred_vertices=pred_vertices,
        mhr_model_params=_as_vector(
            data,
            "mhr_model_params",
            expected_dim=204,
            required=require_mhr_compact,
        ),
    )
