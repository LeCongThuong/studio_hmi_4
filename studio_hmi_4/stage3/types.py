"""Shared dataclasses for stage-3 optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch


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
    w_temporal_velocity: float = 0.0
    w_temporal_accel: float = 0.0
    optimize_hand_pose: bool = True
    w_hand_reg: float = 1e-3
    use_anchor_similarity: bool = True
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
    fixed_hand_pose_params: Optional[np.ndarray] = None
    fixed_scale_params: Optional[np.ndarray] = None
    fixed_shape_params: Optional[np.ndarray] = None
    fixed_expr_params: Optional[np.ndarray] = None


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
