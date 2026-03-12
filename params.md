# params

## 1. How to use this guide

This file focuses on the parameters that matter for actually running and tuning the current pipeline.

The fastest way to use the repo is:

1. use `run_full_pipeline.py` for normal runs,
2. use stage-specific scripts only when debugging,
3. change only high-impact parameters first,
4. treat the rest as advanced or internal controls.

## 2. Minimal required parameters for the main pipeline

These are the real required inputs for [`run_full_pipeline.py`](/home/love_you/Documents/studio_hmi_4/run_full_pipeline.py).

| Parameter | Meaning | Why it matters | Recommended value |
| --- | --- | --- | --- |
| `--image_folder` | Root containing frame folders or one frame folder | Defines the full input sequence | Use extracted frames with layout `<frame>/<cam>.jpg` |
| `--output_root` | Root for all outputs | Keeps stages, summaries, and exports together | One dedicated folder per experiment |
| `--cams` | Camera stems, for example `left front right` | Drives file lookup and camera order | Match image stems exactly |
| `--caliscope_toml` | Camera calibration file | Required for triangulation and reprojection | Use the exact TOML used for the recorded rig |
| `--checkpoint_path` | SAM-3D checkpoint | Required for Stage 1 inference | Use the checkpoint matched to your SAM install |
| `--mhr_path` | Local MHR asset path for Stage 1 | Needed by the SAM model wrapper | Use the SAM/MHR asset bundle that matches the checkpoint |
| `--hf_repo` or `--opt_ckpt` + `--opt_mhr_pt` | Stage 3 model loading source | Required for optimization forward passes | Prefer `--hf_repo facebook/sam-3d-body-dinov3` if your environment supports it |
| `--device` | `cuda` or `cpu` | Affects runtime heavily | Use `cuda` if available |

## 3. Main-runner parameters you are most likely to tune

| Parameter | Default | Suitable range | Why it is important | When the default is suitable | How to tune it |
| --- | --- | --- | --- | --- | --- |
| `--min_views` | `2` | `2-4` | Minimum views needed to process a frame | Use `2` unless you want to skip low-view frames aggressively | Increase only if 2-view triangulation is too unstable |
| `--frame_rel` | unset | one relative frame dir | Useful for debug runs on a single frame | Leave unset for full sequences | Set to one frame id while tuning Stage 2 and 3 |
| `--overwrite` | off | on/off | Controls cache reuse | Off is correct for normal iteration | Turn on only when you changed inputs or code |
| `--enable_specialized_hand_fusion` | off | on/off | Enables WiLoR hand replacement in Stage 1 | Turn on for hand-focused tasks | Compare hand debug overlays before and after |
| `--specialized_hand_input_root` | empty | valid path | Required when using precomputed hand detections | Use only with WiLoR precompute output | Verify file layout matches `<root>/<rel_dir>/<image>.npy` |
| `--fixed_mhr_param_frame_idx` | unset | one clean frame index | Fixes scale/shape across sequence | Strongly recommended for one performer | Choose a clean frontal frame with good detections |
| `--fixed_mhr_param_cam` | `front` | camera name | Chooses which camera provides fixed non-pose params | `front` is usually best | Use the camera with the cleanest person crop |
| `--freeze_lower_body` | off | on/off | Suppresses lower-body jitter in mostly static sequences | Recommended when the performer stands mostly still | Compare lower-body stability in the output meshes |
| `--save_sequence_mp4` | off | on/off | Saves a quick sequence debug animation | Useful for QA runs | Enable on smaller experiments, disable on bulk runs |

## 4. Stage 1 parameters

Relevant script: [`sam3d_inference.py`](/home/love_you/Documents/studio_hmi_4/sam3d_inference.py)

### 4.1 Core Stage 1 controls

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--bbox_thresh` | `0.8` | `0.6-0.9` | Controls detector confidence threshold | Keep `0.8` if the person is clear and single-subject |
| `--person_select_strategy` | `largest_bbox` | `first`, `largest_bbox`, `person_index` | Critical when multiple people appear | Use `largest_bbox` for single main subject |
| `--person_index` | `0` | `0...N-1` | Only used when `person_index` strategy is selected | Use only for controlled multi-person debugging |
| `--use_mask` | off | on/off | Enables mask-conditioned prediction | Keep off unless you have evidence mask conditioning helps your data |
| `--debug` | off | on/off | Saves renders and meshes | Turn on only while diagnosing Stage 1 |
| `--save_mhr_params` | off | on/off | Saves extracted pose/scale/shape/expr sidecars | Useful if you want separate Stage 1 parameter inspection |

### 4.2 Specialized hand fusion controls

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--specialized_hand_source` | `precomputed` | `precomputed`, `live` | Chooses where hand detections come from | Use `precomputed` for stable workflows |
| `--specialized_hand_model` | `none` | `none`, `wilor` | Declares the hand model type | Use `wilor` only when fusion is enabled |
| `--specialized_hand_detector_conf` | `0.3` | `0.2-0.5` | Affects WiLoR hand detection recall vs precision | `0.3` is a balanced start |
| `--specialized_hand_rescale_factor` | `2.5` | `2.0-3.0` | Controls crop enlargement around the hand | `2.5` is suitable for medium framing |
| `--specialized_hand_wrist_max_dist_px` | `140` | `80-180` on 1080p-like imagery | Rejects wrong hand associations | Start with `140`, reduce if wrong hands are being attached |
| `--replace_wrist_with_specialized` | off | on/off | Lets specialized hand replace the wrist itself | Leave off unless the SAM wrist is consistently worse |
| `--specialized_hand_debug_vis` | off | on/off | Writes pre/post fusion overlays | Turn on while tuning hand fusion |

### 4.3 Stage 1 tuning method

Input:

- 20 to 50 representative frames,
- a mix of easy and hard hand poses,
- debug overlays.

Desired output:

- correct subject chosen,
- no obviously wrong hand replacement,
- higher hand detail than SAM-only,
- limited fallback to bad specialized detections.

Best parameter indicator:

- visually correct hand overlays,
- stable `specialized_hand_status_right/left`,
- reasonable `specialized_hand_num_points`,
- fewer downstream bad frames in hand-heavy motion.

## 5. Stage 2 parameters

Relevant script: [`triangulate_mhr3d_gt.py`](/home/love_you/Documents/studio_hmi_4/triangulate_mhr3d_gt.py)

### 5.1 Geometry and coordinate interpretation

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--normalized` | off | on/off | Forces 2D coordinates to be treated as normalized | Keep off if Stage 1 already writes pixel coordinates |
| `--pixel` | off | on/off | Forces 2D coordinates to be treated as pixels | Use when auto-detection is wrong |
| `--invert_extrinsics` | off | on/off | Handles rigs stored with opposite extrinsic convention | Use only if reprojection is obviously mirrored or exploded |

### 5.2 Robust triangulation controls

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--score_type` | `median` | `median`, `trimmed`, `huber` | Chooses candidate scoring rule | `median` is the safest default |
| `--huber_delta` | `10 px` | `5-20 px` | Only used for Huber scoring | Start at `10` if using `huber` |
| `--inlier_thresh` | `30 px` | `15-40 px` | Defines which views are trusted before LM | `30` is reasonable for moderate calibration noise |
| `--robust_lm` | off | on/off | Enables Huber-style weighting inside LM | Turn on if one camera is often noisy |
| `--robust_lm_delta` | `10 px` | `5-20 px` | Residual cutoff for robust LM | Start at `10` |
| `--no_reseed_from_inliers` | off | on/off | Disables DLT reseed from selected inliers | Leave reseed enabled unless it harms a known corner case |

### 5.3 LM refinement controls

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--lm_iters` | `25` | `15-40` | More iterations can refine hard points | `25` is a good balance |
| `--lm_lambda` | `1e-3` | `1e-4` to `1e-2` | Initial damping for LM | Keep default unless LM becomes unstable |
| `--lm_eps` | `1e-4` | `1e-5` to `1e-3` | Finite-difference step scale | Keep default unless numerical Jacobian looks too noisy |

### 5.4 Stage 2 tuning method

Input:

- one or more frames with accurate multi-view correspondences,
- saved debug overlays,
- per-camera refined reprojection errors.

Desired output:

- low refined reprojection error,
- at least two inlier views for most supervised points,
- no large visual mismatch between observed and projected 2D overlays.

Best parameter indicator:

- refined mean reprojection error consistently lower than init error,
- fewer points left as NaN,
- stable triangulated hands without wild point jumps.

## 6. Stage 3 parameters

Relevant script: [`optimize_mhr_pose.py`](/home/love_you/Documents/studio_hmi_4/optimize_mhr_pose.py)

### 6.1 Core optimization controls

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--iters` | `200` | `100-300` | Total optimization steps | `200` is a good starting point |
| `--lr` | `0.05` | `0.01-0.1` | Most sensitive optimization knob | Lower if losses oscillate, raise if convergence is too slow |
| `--with_scale` | off | on/off | Lets similarity alignment estimate scale | Keep off if calibration scale is trusted; turn on when scale mismatch remains |
| `--huber_m` | `0.03 m` | `0.02-0.05 m` | Robustness cutoff for 3D residuals | `0.03` is a sensible default |
| `--min_iters` | `50` | `20-100` | Prevents early stop too soon | Keep `50` unless runs are too short |
| `--early_stop_patience` | `60` | `20-100` | Stops after no improvement | `60` is conservative |
| `--early_stop_tol` | `1e-6` | `1e-7` to `1e-5` | Improvement needed to reset patience | Keep default unless losses are very noisy |
| `--loss_divergence_ratio` | `3.0` | `2.0-5.0` | Aborts diverging runs early | `3.0` is reasonable |

### 6.2 Regularization and temporal controls

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--w_pose_reg` | `1e-3` | `1e-4` to `1e-2` | Keeps pose near initialization | Keep near default unless the optimizer overfits noisy GT |
| `--w_hand_reg` | `1e-3` | `1e-4` to `1e-2` | Keeps hand108 near init | Raise if hand pose becomes unstable |
| `--w_temporal` | `3e-3` | `0` to `1e-2` | Smooths frame-to-frame pose | Good for sequences, unnecessary for single isolated frames |
| `--temporal_init_blend` | `0.7` | `0.4-0.9` | Controls how strongly temporal history shapes initialization | `0.7` is suitable when history is mostly reliable |

### 6.3 Reliability and masking controls

| Parameter | Default | Suitable range | Why it matters | Recommended use |
| --- | --- | --- | --- | --- |
| `--no_optimize_hand_pose` | off | on/off | Disables hand pose optimization | Leave off for hand-focused sequences |
| `--no_anchor_similarity` | off | on/off | Uses all points, not just stable upper-body anchors, for alignment | Leave anchor similarity on unless you have a strong reason |
| `--bad_loss_threshold` | `3e-5` | data-dependent | Defines "bad" total loss | Keep default first, then calibrate from your own summary stats |
| `--bad_data_loss_threshold` | `2e-5` | data-dependent | Defines "bad" data-fit loss | Same strategy as above |
| `--bad_loss_growth_ratio` | `1.5` | `1.2-2.0` | Detects late divergence | `1.5` is a reasonable guardrail |
| `--min_valid_points` | `6` | `4-10` | Minimum valid 3D points needed | Keep `6` unless camera coverage is sparse |
| `--zero_weight_strategy` | `uniform_finite` | `uniform_finite`, `fail` | Controls fallback when weights collapse | `uniform_finite` is safer for production runs |
| `--freeze_lower_body` | off | on/off | Stops lower-body jitter in mostly static performers | Recommended for sign-like upper-body motion |

### 6.4 Stage 3 tuning method

Input:

- triangulated `.npz`,
- loss curves,
- `sequence_summary.json`,
- visual mesh playback.

Desired output:

- low `best_data_loss`,
- low bad-frame rate,
- no obvious body-size drift,
- smooth but not over-smoothed motion,
- no optimization divergence.

Best parameter indicator:

- `best_data_loss` decreases,
- `bad_loss` count decreases,
- recovered-frame fraction stays low,
- hand motion stays responsive while torso and legs remain stable.

## 7. Sequence recovery and smoothing parameters

These live in the full-pipeline config and matter even though most are not exposed in the small public CLI.

| Parameter | Current value | Suitable range | Why it matters |
| --- | --- | --- | --- |
| `max_stale_temporal_frames` | `40` | `10-60` | Disables temporal priors after long bad streaks |
| `max_edge_recovery_copy_span` | `15` | `5-20` | Prevents unbounded copied tails at sequence edges |
| `enable_smoothing` | `True` | on/off | Final sequence polishing |
| `smoothing_alpha` | `0.65` | `0.5-0.8` | EMA smoothing strength |
| `smoothing_median_window` | `5` | `3-7` | Median filter window size |
| `smoothing_outlier_sigma` | `3.5` | `2.5-4.5` | Outlier suppression threshold |
| `sequence_fps` | `20` | `10-30` | Only affects debug MP4 playback |

## 8. Parameter-selection workflow I recommend

### 8.1 Build a small tuning set

Use:

- 20 to 50 representative frames,
- both easy and difficult hand poses,
- at least one short sequence with failures.

### 8.2 Tune in this order

1. Stage 1 person selection and hand fusion.
2. Stage 2 inlier threshold and robust scoring.
3. Stage 3 learning rate and temporal weights.
4. Bad-frame thresholds.
5. Smoothing.

### 8.3 What the input and output of tuning should be

Input:

- raw frames,
- calibration,
- candidate parameter set,
- debug artifacts.

Output:

- lower refined reprojection error,
- lower `bad_loss` rate,
- lower recovery rate,
- cleaner hand overlays,
- stable mesh playback.

### 8.4 Concrete tuning targets

For hand fusion:

- maximize correct specialized replacements,
- minimize visually wrong left/right or off-body replacements.

For triangulation:

- minimize refined reprojection error,
- keep most subset points finite,
- reduce view-to-view disagreement.

For optimization:

- minimize `best_data_loss`,
- minimize divergence,
- keep body dimensions stable across time.

For smoothing:

- reduce jitter without lagging fast hand motion too much.

## 9. Parameters I would not tune first

These exist, but they should stay near current defaults until a test harness exists:

- `lm_lambda`
- `lm_eps`
- `robust_lm_delta`
- `w_pose_reg`
- `w_hand_reg`
- `zero_weight_strategy`
- `smoothing_*`

They are important, but their correct values depend on a more structured benchmark than the current repo provides.
