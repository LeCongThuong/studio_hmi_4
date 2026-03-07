# CLI Simplification Plan (Final Decisions)

## Goal
Reduce argument sprawl and maintenance burden by removing low-value, duplicate, and rarely-used knobs.

This list is based on current pipeline usage (single person, mostly static body, hand-focused motion).

## 1) Remove Immediately (Low Risk)

These are duplicates or mostly debug-only and add noise for normal users.

- `--debug_4d` (duplicate alias of `--debug_sequence`)
- `--save_4d_mp4` (duplicate alias of `--save_sequence_mp4`)
- `--debug_triangulation_every_frame` (very niche, high UI noise)
- `--specialized_hand_verbose` (developer logging only)
- `--topk_print` (debug formatting only)

## 2) Remove from Public CLI (Keep Internal Defaults)

These should be hardcoded in code/config and not exposed to users.

### File naming / plumbing
- `--triangulated_name`
- `--optimized_name`
- `--smoothed_name`
- `--sequence_mp4_name`

### Stage-1 backend internals
- `--detector_name`
- `--segmentor_name`
- `--fov_name`
- `--detector_path`
- `--segmentor_path`
- `--fov_path`
- `--use_mask`

### Specialized-hand internals (for precomputed WiLoR flow)
- `--specialized_hand_source` (fix to `precomputed`)
- `--specialized_hand_model` (fix to `wilor`)
- `--specialized_hand_device` (irrelevant when precomputed)
- `--wilor_pretrained_dir`
- `--wilor_repo_id`
- `--specialized_hand_debug_dirname`

### Triangulation fine-tuning internals
- `--lm_lambda`
- `--lm_eps`
- `--robust_lm_delta`
- `--huber_delta`

### Optimization fine-tuning internals
- `--w_pose_reg`
- `--w_hand_reg`
- `--huber_m`
- `--zero_weight_strategy` (fix to `uniform_finite`)

### Smoothing internals
- `--smoothing_alpha`
- `--smoothing_median_window`
- `--smoothing_outlier_sigma`

## 3) V2 Policy (Committed)

Workflow is fixed to: no retry bad frames, recover by interpolation/copy, hands optimized, anchor similarity on.

- `--bad_frame_max_retries` (remove retry branch entirely)
- `--w_temporal_velocity` (keep 0 internally)
- `--w_temporal_accel` (keep 0 internally)
- `--temporal_extrapolation` (only needed with velocity term)
- `--no_optimize_hand_pose` (always optimize hands)
- `--fixed_hand_pose_from_reference` (v2 says do not freeze hand pose)
- `--no_anchor_similarity` (always use anchor similarity)
- `--no_recover_bad_frames` (always recover; remove toggle)
- `--no_fill_missing_frames` (always fill; remove toggle)

## 4) Strong Candidate Removals (Approved)

These are now accepted for removal from public CLI.

- `--normalized` and `--pixel` (auto-detect coordinate mode)
- `--invert_extrinsics` (keep only if calibration source is inconsistent)
- `--with_scale` (for mostly static single-person, often unnecessary)
- `--person_select_strategy` and `--person_index` (if truly always one person)
- `--replace_wrist_with_specialized` (pick one policy and lock it)
- `--save_opt_debug`, `--debug_inference`, `--debug_triangulation`, `--debug_sequence` (move to a single `--debug` mode)

## 5) Final Minimal Public `run_full_pipeline.py` CLI

Keep only these public user-facing flags:

- `--image_folder`
- `--output_root`
- `--cams`
- `--caliscope_toml`
- `--checkpoint_path`
- `--mhr_path`
- `--hf_repo` (or `--opt_ckpt` + `--opt_mhr_pt`)
- `--device`
- `--frame_rel`
- `--overwrite`
- `--min_views`
- `--enable_specialized_hand_fusion`
- `--specialized_hand_input_root`
- `--fixed_mhr_param_frame_idx`
- `--fixed_mhr_param_cam`
- `--freeze_lower_body`
- `--save_sequence_mp4`

Everything else should be internal defaults or a small number of presets (for example: `--profile fast`, `--profile robust`).

## 6) Decision Log

- Minimal public CLI in Section 5 is accepted.
- V2 policy in Section 3 is accepted.
- All strong candidate removals in Section 4 are accepted.
