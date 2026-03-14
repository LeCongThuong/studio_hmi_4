"""Shared dataclasses for sequence orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class FullPipelineConfig:
    """Configuration for orchestrating all pipeline stages."""

    image_folder: str
    output_root: str
    cams: List[str]
    caliscope_toml: str
    mhr_py: str = "mhr70.py"
    toml_sections: Optional[List[str]] = None
    checkpoint_path: str = ""
    mhr_path: str = ""
    detector_name: str = "vitdet"
    segmentor_name: str = "sam2"
    fov_name: str = "moge2"
    detector_path: str = ""
    segmentor_path: str = ""
    fov_path: str = ""
    bbox_thresh: float = 0.8
    use_mask: bool = False
    debug_inference: bool = True
    save_mhr_params: bool = False
    person_select_strategy: str = "largest_bbox"
    person_index: int = 0
    enable_specialized_hand_fusion: bool = False
    specialized_hand_source: str = "precomputed"
    specialized_hand_model: str = "none"
    specialized_hand_input_root: str = ""
    specialized_hand_device: str = "cuda"
    specialized_hand_detector_conf: float = 0.3
    specialized_hand_rescale_factor: float = 2.5
    specialized_hand_wrist_max_dist_px: float = 140.0
    replace_wrist_with_specialized: bool = False
    specialized_hand_debug_vis: bool = False
    specialized_hand_debug_dirname: str = "specialized_hand_debug"
    specialized_hand_verbose: bool = False
    wilor_pretrained_dir: str = ""
    wilor_repo_id: str = "warmshao/WiLoR-mini"
    frame_rel: Optional[str] = None
    overwrite: bool = False
    skip_inference: bool = False
    skip_triangulation: bool = True
    npy_root: Optional[str] = None
    triangulated_name: str = "triangulated.npz"
    optimized_name: str = "opt_out.npy"
    normalized: bool = False
    pixel: bool = False
    invert_extrinsics: bool = False
    lm_iters: int = 25
    lm_lambda: float = 1e-3
    lm_eps: float = 1e-4
    score_type: str = "median"
    huber_delta: float = 10.0
    inlier_thresh: float = 30.0
    robust_lm: bool = False
    robust_lm_delta: float = 10.0
    save_triangulation_debug: bool = True
    hf_repo: Optional[str] = None
    opt_ckpt: Optional[str] = None
    opt_mhr_pt: str = ""
    device: str = "cuda"
    iters: int = 200
    lr: float = 5e-2
    with_scale: bool = True
    huber_m: float = 0.03
    w_pose_reg: float = 1e-3
    w_hand_reg: float = 1e-3
    w_temporal: float = 3e-3
    temporal_init_blend: float = 0.7
    fixed_mhr_param_frame_idx: Optional[int] = None
    fixed_mhr_param_cam: str = "front"
    fixed_lower_body_pose_frame_idx: Optional[int] = None
    fixed_lower_body_pose_cam: str = "front"
    optimize_hand_pose: bool = True
    use_anchor_similarity: bool = True
    bad_loss_threshold: float = 3e-3
    bad_data_loss_threshold: float = 2.5e-3
    bad_loss_growth_ratio: float = 1.5
    min_valid_points: int = 6
    zero_weight_strategy: str = "uniform_finite"
    freeze_lower_body: bool = False
    topk_print: int = 10
    save_opt_debug: bool = True
    min_views: int = 2
    max_stale_temporal_frames: int = 40
    max_edge_recovery_copy_span: int = 15
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.65
    smoothing_median_window: int = 5
    smoothing_outlier_sigma: float = 3.5
    smoothed_name: str = "opt_out_smoothed.npy"
    save_sequence_mp4: bool = False
    sequence_mp4_name: str = "sequence_debug.mp4"
    sequence_fps: int = 20
    save_summary_json: bool = True


@dataclass
class FramePipelineResult:
    rel_dir: str
    frame_index: Optional[int]
    npy_dir: Optional[Path]
    available_cams: List[str]
    used_cams: List[str]
    triangulated_npz: Optional[Path]
    optimized_npy: Optional[Path]
    smoothed_npy: Optional[Path]
    status: str
    best_loss: Optional[float] = None
    final_loss: Optional[float] = None
    best_data_loss: Optional[float] = None
    final_data_loss: Optional[float] = None
    best_iter: Optional[int] = None
    is_bad_loss: bool = False
    recovered_from: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FullPipelineResult:
    output_root: Path
    npy_root: Path
    frames: List[FramePipelineResult]
    summary_json: Optional[Path] = None


@dataclass
class FrameInputEntry:
    rel_dir: str
    npy_dir: Optional[Path]
    frame_index: Optional[int]
    available_cams: List[str]
