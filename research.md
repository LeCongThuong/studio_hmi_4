# SAM-3D Body Pipeline Deep Research Report

## 1) Scope and Method

I analyzed this repository end-to-end, with focus on:

- `sam3d_inference.py` (Stage 1)
- `triangulate_mhr3d_gt.py` (Stage 2)
- `optimize_mhr_pose.py` (Stage 3)
- `run_full_pipeline.py` orchestration
- sequence repair/smoothing (`video_temporal_utils.py`)

I also inspected upstream SAM-3D Body source code directly from Meta's `facebookresearch/sam-3d-body` repository (local clone in `/tmp/sam-3d-body-src-2`) to verify internals of SAM3D inference + Momentum Human Rig (MHR) parameterization.

I additionally built synthetic, controlled tests around `run_optimization` (with a mocked MHR head) to validate failure modes with concrete evidence.

---

## 2) What This Project Does (High-Level)

This project is a 3-stage multi-view reconstruction/fitting pipeline:

1. Stage 1 (`sam3d_inference.py`)
- Runs SAM-3D Body per camera image.
- Saves per-view dictionaries to `.npy`.

2. Stage 2 (`triangulate_mhr3d_gt.py`)
- Reads per-view 2D keypoints (`pred_keypoints_2d`), selects an MHR subset (hands + shoulders/elbows/wrists), triangulates 3D points, and refines with per-point LM.
- Saves a `.npz` bundle (`points3d_refined`, `inlier_mask`, reprojection diagnostics).

3. Stage 3 (`optimize_mhr_pose.py`)
- Loads Stage 2 3D points as pseudo-GT.
- Chooses best camera initialization from per-view SAM outputs.
- Optimizes `body_pose_params` (133D) to align MHR FK keypoints to triangulated subset via robust loss + optional temporal priors.

`run_full_pipeline.py` orchestrates this over single frames or sequences, with optional retries, recovery interpolation, smoothing, and summary output.

---

## 3) Stage 1 Deep Dive: SAM3D Inference

## 3.1 Local wrapper behavior (`sam3d_inference.py`)

Key behavior:

- Builds estimator from `load_sam_3d_body(...)` and `SAM3DBodyEstimator`.
- Optional detector, segmentor, FOV estimator.
- For each image:
  - `outputs = estimator.process_one_image(...)`
  - `extract_primary_output(outputs)` picks one mapping (details in bugs section)
  - saves `<output>/npy/<rel_dir>/<stem>.npy`
  - optional debug renders/mesh
  - optional MHR param sidecar (`body_pose_params`, `hand_pose_params`, `scale_params`, `shape_params`, `expr_params`)

## 3.2 Upstream estimator behavior (verified)

From upstream `sam_3d_body/sam_3d_body_estimator.py`:

- `process_one_image`:
  - detects boxes if detector provided, else full image box
  - optional mask via SAM segmentor
  - builds crop batch and runs `model.run_inference`
  - returns list of per-person dictionaries

Per-person output keys include:

- `pred_keypoints_2d`, `pred_keypoints_3d`
- `pred_vertices`, `pred_joint_coords`, `pred_global_rots`
- `body_pose_params`, `hand_pose_params`, `scale_params`, `shape_params`, `expr_params`
- `pred_cam_t`, `focal_length`, etc.

## 3.3 Important upstream geometric details

From upstream `PerspectiveHead` and `MHRHead`:

- camera head predicts `pred_cam = (s, tx, ty)`
- internally flips camera convention on `s` and `ty`
- depth is reconstructed as `tz = 2*f / (bbox_size*s*default_scale_factor + eps)`
- projected `pred_keypoints_2d` are in full-image pixel coordinates

From upstream `MHRHead.forward`:

- output verts/keypoints/joints have `y,z` sign flip to match camera system
- output keypoints are reduced from 308 to first 70 for `pred_keypoints_3d`

This explains why local Stage 3 uses `apply_repo_camera_flip_xyz(...)` for consistency when calling `mhr_forward` directly.

---

## 4) Momentum Human Rig (MHR) Parameterization (Deep)

From upstream `mhr_head.py` and `mhr_utils.py`:

## 4.1 Core dimensions

- Body continuous representation: `body_cont_dim = 260`
- Body model parameter representation: `body_pose_params = 133`
- Hand latent per side: `54`, total `108`
- Shape: `45`
- Scale latent: `28` (mapped to 68 actual scales)
- Face expression: `72`

`npose` token regressed by head is:

- `6 (global rot 6D) + 260 (body cont) + 45 + 28 + 108 + 72 = 519`

## 4.2 260 -> 133 mapping

`compact_cont_to_model_params_body` maps:

- 3-DoF rotations represented in 6D continuous form
- 1-DoF rotations represented in sin/cos 2D form
- plus 6 translational DoFs at the tail of the 133 vector

So body params are heterogeneous: not all joints are plain XYZ Euler.

## 4.3 Hand masking in this repository

Upstream defines `mhr_param_hand_mask` over 133 body params (indices 62..115 are hand-related in model-param space).

Local optimization freezes:

- all hand-related body dims (`mhr_param_hand_mask`)
- last 3 dims (treated as jaw/non-target in this repo)

Meaning: local Stage 3 intentionally optimizes a reduced subset of body params and keeps hands/jaw fixed or externally controlled.

## 4.4 Why local `mhr_fk` tries both pose sizes

Local `mhr_fk` tries `body_pose_params[:130]` then full 133.

Reason:

- upstream `mhr_forward` explicitly slices `body_pose_params[..., :130]` internally
- local fallback is defensive for possible checkpoint/API variants

---

## 5) Stage 2 Deep Dive: `triangulate_mhr3d_gt.py`

## 5.1 Subset selection

Subset =

- right hand keypoints (20)
- left hand keypoints (20)
- shoulders/elbows/wrists (6)

Total expected `M=46` subset points.

Names are resolved using `pose_info` in `mhr70.py`.

## 5.2 Input contract

Per camera file (`<cam>.npy`/`.npz`) must contain `pred_keypoints_2d` as:

- `(70,2)` or `(N,70,2)`

Triangulation runs on subset indices only.

## 5.3 Camera model

Caliscope camera fields expected:

- intrinsics `matrix` (K)
- distortion `distortions` (D)
- extrinsics `rotation` (rvec), `translation` (tvec)
- image `size=[w,h]`

Pipeline uses:

- undistort to normalized coords
- DLT triangulation with `P_norm = [R|t]`
- reprojection with full distorted projection for pixel-space errors

## 5.4 Robust triangulate + BA flow

Per keypoint `j`:

1. Pairwise DLT candidate from camera pairs.
2. Score candidate by reprojection errors across valid views using robust score:
- `median`, `trimmed`, or `huber`.
3. Inlier view gating by pixel threshold (`inlier_thresh`, default 30 px).
4. Optional reseed from inliers.
5. LM refinement per point (optionally robust Huber-weighted residuals in LM).

Outputs:

- `points3d_init`, `points3d_refined`
- `best_pair_idx`
- `inlier_mask` (M,V)
- per-cam reprojections + error arrays + means

## 5.5 Intricacy that matters downstream

If a keypoint has <2 valid observations or degenerates, triangulation leaves that point as `NaN` in `points3d_refined`.

This is a valid Stage-2 behavior, but Stage-3 must robustly mask invalid points (currently it does not; see critical bugs).

---

## 6) Stage 3 Deep Dive: `optimize_mhr_pose.py`

## 6.1 Initialization and view selection

- Loads Stage-2 NPZ.
- For each camera view `.npy`, runs FK once, extracts 70 keypoints, takes subset.
- Computes weighted Umeyama similarity to `points3d_refined`.
- Scores per-view 3D residual; picks best init camera (tie-break by mean reproj px from NPZ).

## 6.2 Objective and optimization strategy

Per iteration:

1. FK from current pose.
2. Compute rigid/similarity alignment to GT subset (`Umeyama`) with no gradient through alignment (alternating scheme).
3. Data loss: weighted Huber on aligned subset residuals.
4. Regularization terms:
- pose-to-init (`w_pose_reg`)
- temporal prior (`w_temporal`)
- temporal velocity (`w_temporal_velocity`)
- temporal acceleration (`w_temporal_accel`)
5. Adam update with gradient clipping and masked dimensions frozen.

Early stop conditions:

- patience on non-improvement
- divergence ratio stop

After loop:

- restores best pose
- re-runs full FK to write final aligned geometry and diagnostics

## 6.3 Output contract

Saves `out_npy` as SAM-style dict + optimization diagnostics:

- updated `body_pose_params`
- `pred_keypoints_3d`, `pred_vertices`, `pred_joint_coords`, `pred_global_rots`
- many `opt_*` metrics (loss history, best/final losses, flags)

---

## 7) Confirmed Bugs (with evidence)

The following are high-confidence bugs confirmed by code + reproduction.

## BUG-1 (Critical): NaN/Inf in triangulated GT can crash optimization

Where:

- `optimize_mhr_pose.py` lines around loading `points3d_refined` and calling `umeyama_similarity`.
- Code refs: `optimize_mhr_pose.py:534`, `optimize_mhr_pose.py:581`, `optimize_mhr_pose.py:673`, `optimize_mhr_pose.py:773`.

Root cause:

- Stage 2 can legitimately output NaN points.
- Stage 3 uses all points directly (no finite-point mask).
- `torch.linalg.svd` in Umeyama fails on non-finite covariance.

Observed reproduction:

- Synthetic test with one NaN keypoint produced:
  - `_LinAlgError: linalg.svd ... input matrix contained non-finite values`

Impact:

- Frame optimization fails hard (`optimization_failed`) despite partially usable triangulation.

## BUG-2 (Critical): Zero inlier weights creates false-perfect loss and false “good” frame

Where:

- `optimize_mhr_pose.py` weighted loss/scoring using `wM` from `inlier_mask`.
- Code refs: `optimize_mhr_pose.py:538`, `optimize_mhr_pose.py:541`, `optimize_mhr_pose.py:584`, `optimize_mhr_pose.py:679`, `optimize_mhr_pose.py:781`.

Root cause:

- If `inlier_mask` is all zeros (or effectively all zero weights), then:
  - view score becomes 0
  - data loss becomes 0
  - optimization appears perfect without fitting data
- No guard exists for `sum(wM)==0`.

Observed reproduction:

- Synthetic test with large GT mismatch (>2m mean residual) and zero inlier mask yielded:
  - `best_loss=0`, `best_data_loss=0`, `is_bad=False`

Impact:

- Pipeline can silently accept garbage fit as high quality.

## BUG-3 (High): `final_loss` used for bad-frame classification is stale (pre-restore), causing false bad flags

Where:

- `optimize_mhr_pose.py`: `final_loss` taken from last iteration before best-pose restore; later classification uses this stale value.
- Code refs: `optimize_mhr_pose.py:748`, `optimize_mhr_pose.py:752`, `optimize_mhr_pose.py:848`, `optimize_mhr_pose.py:864`, `optimize_mhr_pose.py:885`.

Root cause:

- Code restores `best_pose` before output geometry.
- But `final_loss` remains from the last optimization step (which may be worse).
- `classify_bad_optimization` uses growth ratio `final_loss / best_loss`, so frame can be marked bad even when exported pose is best.

Observed reproduction:

- Synthetic run: `best_loss=0.000511`, `final_loss=0.001197`, `best_iter=3`, `is_bad=True`.
- Exported geometry came from restored best pose.

Impact:

- False bad-frame retries/recoveries, unnecessary interpolation/copy fallback, quality instability.

## BUG-4 (Critical): Sequence smoothing propagates NaNs across neighboring frames

Where:

- `video_temporal_utils.py` `smooth_frame_dict_sequence`.
- Code refs: `video_temporal_utils.py:153`, `video_temporal_utils.py:160`, `video_temporal_utils.py:165`, `video_temporal_utils.py:166`, `video_temporal_utils.py:168`.

Root cause:

- Frame validity is based on key existence/shape, not finite values.
- NaN-containing frames are treated as valid samples.
- median/outlier/EMA steps are not NaN-safe (`np.median`, arithmetic), causing contamination.

Observed reproduction:

- Input: finite, NaN, finite sequence.
- Output after smoothing: all frames NaN.

Impact:

- A single corrupted frame can poison the entire smoothed sequence.

## BUG-5 (High, interface mismatch): Stage 2 NaN semantics are not handled by Stage 3 contract

Where:

- Stage 2 intentionally emits NaNs for untriangulable points.
- Stage 3 assumes dense finite GT subset.
- Code refs: `triangulate_mhr3d_gt.py:592`, `triangulate_mhr3d_gt.py:715`, `optimize_mhr_pose.py:534`, `optimize_mhr_pose.py:546`.

Root cause:

- Missing contract enforcement/checkpoint between stages.
- No minimum finite-keypoint threshold before optimization.

Impact:

- Intermittent failures and unstable quality depending on occlusion/view availability.

---

## 8) Additional High-Risk Issues (Not all hard-crash, but quality-degrading)

## ISSUE-A: First-person-only selection in Stage 1 can cause identity switching

Where:

- `sam3d_inference.py` `extract_primary_output(outputs)` takes first prediction from list.
- Code refs: `sam3d_inference.py:245`, `sam3d_inference.py:253`, `sam3d_inference.py:346`.

Risk:

- Multi-person scenes or detector ordering changes can switch target person across cameras/frames.

Impact:

- Triangulation and optimization may run but fit inconsistent subject.

## ISSUE-B: `.npy` plain-array loading path in triangulation is brittle

Where:

- `triangulate_mhr3d_gt.py` `load_pred_keypoints_2d` uses `.item()` for any `.npy` ndarray.
- Code refs: `triangulate_mhr3d_gt.py:152`, `triangulate_mhr3d_gt.py:154`.

Risk:

- A valid plain `(70,2)` `.npy` would raise due `.item()` misuse.

Impact:

- avoidable input compatibility failures.

## ISSUE-C: `run_full_pipeline` Stage-1 reuse can pick stale outputs for `frame_rel`

Where:

- `run_full_pipeline.py` reuse check only verifies directory existence + minimum camera files.
- Code refs: `run_full_pipeline.py:681`, `run_full_pipeline.py:688`, `run_full_pipeline.py:691`.

Risk:

- If `output_root` reused across sessions, stale Stage-1 files may be reused for different images.

Impact:

- pipeline runs against wrong data silently.

## ISSUE-D: File matching fallback in optimization may select wrong camera file

Where:

- `find_npy_for_cam` fallback `*{cam}*.npy`.
- Code refs: `optimize_mhr_pose.py:192`, `optimize_mhr_pose.py:196`.

Risk:

- overlapping camera names or auxiliary files can be mis-selected when exact `<cam>.npy` missing.

Impact:

- wrong initialization source and degraded fit.

---

## 9) Parameter Meaning Reference (Practical)

## 9.1 Stage 1 (`sam3d_inference.py`)

- `checkpoint_path`: SAM-3D Body checkpoint (`model.ckpt`)
- `mhr_path`: MHR model asset (`mhr_model.pt`)
- `detector_name`/`detector_path`: human detector backend + weights
- `segmentor_name`/`segmentor_path`: segmentation backend + weights
- `fov_name`/`fov_path`: FOV estimator
- `bbox_thresh`: detector confidence threshold
- `use_mask`: use segmentation-guided inference
- `include_rel_dirs`: process only selected frame subfolders

## 9.2 Stage 2 (`triangulate_mhr3d_gt.py`)

- `normalized`: force treat input 2D as [0,1]
- `pixel`: force treat input as pixel coords
- `invert_extrinsics`: invert rvec/tvec if calibration convention is opposite
- `lm_iters`, `lm_lambda`, `lm_eps`: LM refinement controls
- `score_type`: candidate selection metric (`median`/`trimmed`/`huber`)
- `huber_delta`: robust scoring delta (px)
- `inlier_thresh`: per-view inlier gate threshold (px)
- `robust_lm`, `robust_lm_delta`: robust LM weighting and delta
- `no_reseed_from_inliers`: disable inlier-DLT reseed before LM

## 9.3 Stage 3 (`optimize_mhr_pose.py`)

- `with_scale`: allow similarity scale in Umeyama; otherwise rigid only
- `huber_m`: Huber delta in meters for 3D residual
- `w_pose_reg`: regularize pose toward initialization
- `w_temporal`: regularize toward previous pose
- `w_temporal_velocity`: regularize toward extrapolated velocity target
- `w_temporal_accel`: smooth acceleration in pose space
- `temporal_init_blend`: blend between per-view init and temporal target
- `temporal_extrapolation`: velocity extrapolation gain (prev-prev)
- `bad_loss_threshold`, `bad_data_loss_threshold`: absolute quality gates
- `bad_loss_growth_ratio`: instability gate on final/best ratio

---

## 10) Why You See “Runs but Bad Result”

The most plausible dominant causes in current codebase are:

1. zero/invalid inlier weighting can produce near-zero optimization loss while actual geometry is poor (BUG-2)
2. stale final-loss classification can incorrectly mark good frames bad, triggering recovery interpolation/copy over actual optimization output (BUG-3)
3. NaN values from sparse triangulation can either crash optimization or poison downstream smoothing (BUG-1 + BUG-4)
4. Stage-1 first-person selection can drift subject identity in multi-person or ambiguous detections (ISSUE-A)

This directly matches intermittent behavior: sometimes acceptable, sometimes clearly wrong, without deterministic crashes.

---

## 11) Recommended Fix Order

## Priority 0 (must fix first)

1. In Stage 3, build finite mask before Umeyama/loss:
- keep points where both `gtM` and `predM` are finite
- require minimum valid points (e.g., >= 6) else mark frame unusable

2. Guard zero-weight case:
- if `wM.sum() <= eps`, fail frame (or fallback to uniform weights on finite points)

3. Recompute `final_loss` after restoring best pose, or classify with best-only metrics

4. Make smoothing NaN-safe:
- valid mask should require finite values
- use `nanmedian`/finite-aware operations
- do not let NaN frames act as anchors

## Priority 1

5. Add explicit Stage-2 -> Stage-3 contract checks:
- number of finite subset points
- per-point finite/inlier summary in NPZ
- fail early with clear status

6. Stabilize Stage-1 person selection:
- choose by box area/center consistency across cams/time, not just first list entry

## Priority 2

7. Remove brittle `*{cam}*.npy` fallback or constrain it by strict naming policy.

8. Tighten Stage-1 reuse criteria (hash or timestamp checks against input frame dir).

---

## 12) Validation Artifacts Run During This Research

I executed synthetic tests (mock runtime) that confirmed:

- NaN GT triggers SVD failure in optimization.
- All-zero inlier weights produce zero loss with large residual error and `is_bad=False`.
- Final/best loss mismatch can mark frame bad despite best-pose export.
- Smoothing propagates one NaN frame into all frames.

I could not run full triangulation integration tests in this environment because `cv2` is not installed here, but code-level analysis and optimization-side reproductions already expose the major failure paths.

---

## 13) Final Assessment

The repository design is strong and modular, but there is a critical robustness gap at the Stage-2/Stage-3 interface and in sequence post-processing. The main failure pattern is not a single bug; it is a chain:

- sparse/invalid triangulation points -> optimizer not finite-safe -> bad classification instability -> recovery/smoothing amplifies corruption.

Fixing the four Priority-0 items should materially reduce intermittent bad outputs.
