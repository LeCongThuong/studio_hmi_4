# SAM 3D Body Full Pipeline

This repo is now organized as a 3-stage pipeline:

1. `sam3d_inference.py`: SAM-3D inference per image, save `.npy` outputs (and optional MHR param sidecars).
2. `triangulate_mhr3d_gt.py`: multi-view triangulation + BA to build 3D GT subset.
3. `optimize_mhr_pose.py`: optimize MHR body pose against triangulated 3D GT.

You can run all stages with one command using `run_full_pipeline.py`, or run each stage separately for debugging.

## Input Layout

`run_full_pipeline.py` expects camera image names as stems, for example:

`left.jpg`, `front.jpg`, `right.jpg`

Two supported layouts:

1. Single frame folder:
`<image_folder>/left.jpg`, `<image_folder>/front.jpg`, `<image_folder>/right.jpg`
2. Multi-frame folders:
`<image_folder>/<frame_id>/left.jpg`, `<image_folder>/<frame_id>/front.jpg`, `<image_folder>/<frame_id>/right.jpg`

Recommended video layout:

`input_frames/<k>/<cam>.jpg` where `k` is `0,1,2,...` and each frame can have variable camera count.

The `--cams` names must match these stems.

## One-Command Full Pipeline

```bash
python run_full_pipeline.py \
  --image_folder /path/to/frames_root \
  --output_root /path/to/pipeline_out \
  --cams left front right \
  --caliscope_toml /path/to/config.toml \
  --mhr_py mhr70.py \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --segmentor_name sam3 \
  --hf_repo facebook/sam-3d-body-dinov3 \
  --with_scale
```

### Useful Flags

- `--frame_rel 100`: run only one frame subfolder under inferred `npy` root.
- `--skip_inference --npy_root /path/to/existing/npy`: reuse stage-1 outputs.
- `--skip_triangulation`: reuse existing triangulation files under `output_root/triangulation`.
- `--skip_optimization`: stop after triangulation.
- `--save_mhr_params`: save extracted MHR params from stage-1 under `inference/mhr_params`.
- `--person_select_strategy largest_bbox`: choose stage-1 person selection mode (`first`, `largest_bbox`, `person_index`).
- `--person_index 0`: used only with `--person_select_strategy person_index`.
- `--save_triangulation_debug`: save overlay debug images for triangulation.
- `--debug_inference` / `--debug_triangulation`: interactive/debug rendering for stage-1/stage-2.
- For multi-frame runs, `--debug_triangulation` opens interactive 3D only on the first frame by default.
- `--debug_triangulation_every_frame`: force interactive 3D popup on every frame.
- `--min_views 2`: minimum available views for each frame optimization.
- `--bad_loss_threshold 3e-5 --bad_data_loss_threshold 2e-5`: stricter bad-frame gate for non-human pose prevention.
- `--bad_frame_max_retries 2`: retry bad frames with stronger temporal constraints.
- `--min_valid_points 6 --zero_weight_strategy uniform_finite`: robust stage-3 valid-point/weight controls.
- `--freeze_lower_body`: lock lower-body dimensions to initialization to reduce stationary-leg jitter.
- `--smoothing_alpha 0.65 --smoothing_median_window 5 --smoothing_outlier_sigma 3.5`: sequence smoothing controls.
- `--debug_4d --save_4d_mp4`: interactive 4D playback + MP4 export aliases.

## Output Structure

`<output_root>/`

- `inference/`
- `inference/npy/...` per-camera SAM outputs
- `inference/stage1_meta.json` stage-1 cache contract metadata
- `inference/render/...` and `inference/mesh/...` if `--debug_inference`
- `inference/mhr_params/...` if `--save_mhr_params`
- `triangulation/.../triangulated.npz`
- `triangulation/.../debug/` if `--save_triangulation_debug`
- `optimization/.../opt_out.npy`
- `optimization/.../opt_out_smoothed.npy` (if smoothing enabled)
- `optimization/.../debug_opt/`
- `sequence_summary.json`
- `sequence_debug.mp4` (if `--save_sequence_mp4` or `--save_4d_mp4`)

## Run Stages Manually (Debug)

### 0) Optional: Extract Per-Tick Frames

```bash
python3 extract_per_tick_frames_from_csv.py \
  --root-dir /path/to/session_root \
  --out /path/to/frames_out \
  --cams all
```

Extractor now also writes:
- `frames_index.csv`: per-image row index.
- `frames_manifest.csv`: per-timeframe camera availability (`id`, `num_views`, `cams_present`, `missing_cams`).

### 1) Stage 1: SAM Inference

```bash
python sam3d_inference.py \
  --image_folder /path/to/frame_or_frames_root \
  --output_folder /path/to/stage1_out \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --segmentor_name sam3 \
  --person_select_strategy largest_bbox \
  --debug \
  --save_mhr_params
```

### 2) Stage 2: Triangulation + BA

```bash
python triangulate_mhr3d_gt.py \
  --mhr_py mhr70.py \
  --caliscope_toml /path/to/config.toml \
  --cams left front right \
  --npy_dir /path/to/stage1_out/npy/<frame_rel_or_root> \
  --img_dir /path/to/frame_image_dir \
  --out_npz /path/to/triangulated.npz \
  --debug \
  --debug_dir /path/to/debug_tri
```

### 3) Stage 3: Optimize MHR Pose

```bash
python optimize_mhr_pose.py \
  --npz /path/to/triangulated.npz \
  --npy_dir /path/to/stage1_out/npy/<frame_rel_or_root> \
  --cams left front right \
  --hf_repo facebook/sam-3d-body-dinov3 \
  --with_scale \
  --iters 200 \
  --lr 0.05 \
  --min_valid_points 6 \
  --zero_weight_strategy uniform_finite \
  --freeze_lower_body \
  --bad_loss_threshold 3e-5 \
  --bad_data_loss_threshold 2e-5 \
  --debug_dir /path/to/debug_opt \
  --out_npy /path/to/opt_out.npy
```

If not using `--hf_repo`, use:

```bash
--ckpt /path/to/model.ckpt --mhr_pt /path/to/mhr_model.pt
```

For `run_full_pipeline.py`, the equivalent flags are:

```bash
--opt_ckpt /path/to/model.ckpt --opt_mhr_pt /path/to/mhr_model.pt
```

## Python Module Usage

You can import each stage directly:

- `sam3d_inference.py`: `Demo2Config`, `run_demo`
- `triangulate_mhr3d_gt.py`: `TriangulationConfig`, `run_triangulation`
- `optimize_mhr_pose.py`: `OptimizationConfig`, `run_optimization`
- `run_full_pipeline.py`: `FullPipelineConfig`, `run_full_pipeline`

## Troubleshooting

- `RuntimeError: Insufficient valid weighted points ...`:
  - Increase available views.
  - Check stage-2 triangulation validity for the frame.
  - Lower `--min_valid_points` only if your view coverage is consistently sparse.
- `RuntimeError: freeze_lower_body enabled but no hardcoded lower-body index table for pose_dim=...`:
  - Disable `--freeze_lower_body` for that checkpoint/output format, or
  - add the corresponding hardcoded table for that `pose_dim` in `optimize_mhr_pose.py`.
