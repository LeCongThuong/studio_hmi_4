# SAM 3D Body Full Pipeline

This repo is now organized as a 3-stage pipeline:

1. `sam3d_inference.py`: SAM-3D inference per image, save `.npy` outputs (and optional MHR param sidecars).
2. `triangulate_mhr3d_gt.py`: multi-view triangulation + BA to build 3D GT subset.
3. `optimize_mhr_pose.py`: optimize MHR body pose against triangulated 3D GT.

You can run all stages with one command using `run_full_pipeline.py`, or run each stage separately for debugging.

## Export Final MHR-Ready Results

After pipeline optimization, export one frame-per-folder compact result with:
- `mhr_params.npy` (dict),
- `mhr_params.npz` (arrays),
- `mesh.ply` (optional regenerated mesh),
- `debug_mesh.png` (optional preview).

Frame folder names are preserved from optimization output (for example `0/`, `1/`, ...).

```bash
python export_mhr_final_results.py \
  --optimization_root /path/to/pipeline_out/optimization \
  --output_root /path/to/final_mhr_export \
  --npy_name opt_out_smoothed.npy \
  --fallback_npy_name opt_out.npy \
  --hf_repo facebook/sam-3d-body-dinov3 \
  --device cuda \
  --debug_vis
```

Useful options:
- `--keep_bad`: export frames marked bad-loss too (default skips them).
- `--no_run_mhr_forward`: export params only; do not regenerate mesh through MHR forward.
- `--no_decomposed_params`: export compact-only fields (`mhr_model_params`, `shape_params`, `expr_params`).

### Regenerate Mesh With Official MHR Repo

If you want mesh generation from the official MHR implementation (not SAM head wrapper),
run this companion step from exported `mhr_params.npz`:

```bash
python run_mhr_repo_from_export.py \
  --export_root /path/to/final_mhr_export \
  --mhr_repo_root /path/to/MHR \
  --assets_dir /path/to/MHR/assets \
  --device cuda \
  --debug_vis
```

Per frame output (same frame folder structure):
- `mesh_mhr_repo.ply`
- `debug_mesh_repo.png` (if `--debug_vis`)

Useful options:
- `--output_root /path/to/mhr_repo_meshes`: write meshes to another root.
- `--skip_bad`: skip frames where `is_bad_loss != 0`.
- `--params_name mhr_params.npz`: change param filename if needed.
- `--overwrite`: force regenerate existing meshes.

## V2 Hand-Focused Pipeline (SAM + WiLoR-mini)

For mostly static single-person sequences (for example sign language), use the v2 strategy in `plan_v2.md`:

1. keep SAM-3D-Body as the base 70x2 keypoint source,
2. run WiLoR-mini for stronger hand keypoints,
3. map MANO/OpenPose hand indices to MHR70 hand indices using `hand_mapping.md`,
4. replace only hand keypoints when specialized hand confidence is reliable,
5. triangulate from mixed 2D points (specialized hands + SAM body),
6. optimize both body pose and hand pose (`hand108`), while fixing identity-like parameters per sequence.

Implementation status: this section documents the target v2 update path; the concrete implementation checklist is tracked in `plan_v2.md`.

Important policy in v2:
- alignment transform should use stable upper-body anchors only (shoulders/elbows/wrists),
- `L_vel` and `L_acc` are removed to reduce runtime,
- bad frames are not retried; they are interpolated in recovery.

### WiLoR-mini keypoint extraction notes

Repo checked: `https://github.com/warmshao/WiLoR-mini`.

- Pipeline class: `WiLorHandPose3dEstimationPipeline`.
- Per detected hand output is at `detect_rets[i]["wilor_preds"]`.
- Use:
  - `wilor_preds["pred_keypoints_2d"]` for 2D fusion,
  - `is_right` to map to left/right MHR hand indices.
- WiLoR-mini MANO/OpenPose order:
  - `mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]`.

Use `hand_mapping.md` directly for conversion into MHR70 hand keypoint indices.

### Run WiLoR Separately (Recommended)

If WiLoR cannot be installed in your SAM-3D environment, run it separately and export hand detections first.

1. In WiLoR environment, precompute hand detections:

```bash
python run_wilor_precompute.py \
  --image_folder /path/to/frames_root \
  --output_root /path/to/wilor_precomputed \
  --device cuda \
  --hand_conf 0.3 \
  --rescale_factor 2.5 \
  --debug_vis
```

Optional debug output location:
- `--debug_vis_root /path/to/wilor_debug_vis` (default is `<output_root>/debug_vis`)

2. In SAM-3D pipeline environment, fuse precomputed hands:

```bash
python run_full_pipeline.py \
  --image_folder /path/to/frames_root \
  --output_root /path/to/pipeline_out \
  --cams left front right \
  --caliscope_toml /path/to/config.toml \
  --mhr_py mhr70.py \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --hf_repo facebook/sam-3d-body-dinov3 \
  --enable_specialized_hand_fusion \
  --specialized_hand_source precomputed \
  --specialized_hand_model wilor \
  --specialized_hand_input_root /path/to/wilor_precomputed \
  --specialized_hand_debug_vis
```

Precomputed hand files are expected as:
`<specialized_hand_input_root>/<rel_dir>/<image_stem>.npy` (or `.npz`/`.json`).

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
- `--specialized_hand_debug_vis --specialized_hand_debug_dirname specialized_hand_debug`: save stage-1 overlays comparing SAM hand points (before) vs specialized replacements (after).
- `--save_triangulation_debug`: save overlay debug images for triangulation.
- `--debug_inference` / `--debug_triangulation`: interactive/debug rendering for stage-1/stage-2.
- For multi-frame runs, `--debug_triangulation` opens interactive 3D only on the first frame by default.
- `--debug_triangulation_every_frame`: force interactive 3D popup on every frame.
- `--min_views 2`: minimum available views for each frame optimization.
- `--bad_loss_threshold 3e-5 --bad_data_loss_threshold 2e-5`: stricter bad-frame gate for non-human pose prevention.
- `--bad_frame_max_retries 2`: retry bad frames with stronger temporal constraints.
- `--max_stale_temporal_frames 40`: disable temporal priors after long non-good streaks to avoid stale-pose lock-in.
- `--max_edge_recovery_copy_span 15`: cap one-sided recovery copy distance so long bad tails are not flattened to one repeated pose.
- `--fixed_mhr_param_frame_idx <idx> --fixed_mhr_param_cam front`: lock non-pose MHR params (`hand/scale/shape/expr`) to one reference frame+camera across the sequence (useful for single-person videos to avoid body-size drift).
- Reused optimization files that were recovered now keep a recovered status in summaries (not `ok`) and no longer report copied loss metrics as if they were fresh optimization results.
- `--min_valid_points 6 --zero_weight_strategy uniform_finite`: robust stage-3 valid-point/weight controls.
- `--freeze_lower_body`: lock lower-body dimensions and (for sequence runs) reuse previous-frame similarity alignment for lower-body world stability.
- `--smoothing_alpha 0.65 --smoothing_median_window 5 --smoothing_outlier_sigma 3.5`: sequence smoothing controls.
- `--debug_4d --save_4d_mp4`: interactive 4D playback + MP4 export aliases.

### Recommended settings for hand-focused mostly-static sequences

- Keep non-pose identity fixed:
  - `--fixed_mhr_param_frame_idx <clean_frame>`
  - `--fixed_mhr_param_cam front`
- Keep lower body stable:
  - `--freeze_lower_body`
- Keep interpolation enabled for failure recovery:
  - keep sequence recovery on,
  - use bounded one-sided copy (`--max_edge_recovery_copy_span` small).
- Follow `plan_v2.md` for the no-retry bad-frame policy and specialized hand fusion path.

## Output Structure

`<output_root>/`

- `inference/`
- `inference/npy/...` per-camera SAM outputs
- `inference/stage1_meta.json` stage-1 cache contract metadata
- `inference/render/...` and `inference/mesh/...` if `--debug_inference`
- `inference/specialized_hand_debug/...` (or custom `--specialized_hand_debug_dirname`) if `--specialized_hand_debug_vis`
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
