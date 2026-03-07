# Detailed Pipeline Notes

This file documents the current cleaned codebase, not the old monolithic version.
It keeps Markdown + LaTeX-style notation for technical explanation.

## 1. Current codebase layout

The public entrypoints are still the root scripts:

- [`sam3d_inference.py`](/home/love_you/Documents/studio_hmi_4/sam3d_inference.py)
- [`triangulate_mhr3d_gt.py`](/home/love_you/Documents/studio_hmi_4/triangulate_mhr3d_gt.py)
- [`optimize_mhr_pose.py`](/home/love_you/Documents/studio_hmi_4/optimize_mhr_pose.py)
- [`run_full_pipeline.py`](/home/love_you/Documents/studio_hmi_4/run_full_pipeline.py)
- [`export_mhr_final_results.py`](/home/love_you/Documents/studio_hmi_4/export_mhr_final_results.py)
- [`run_mhr_repo_from_export.py`](/home/love_you/Documents/studio_hmi_4/run_mhr_repo_from_export.py)

The real implementation now lives in:

- [`studio_hmi_4/common/`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/common)
- [`studio_hmi_4/stage1/`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage1)
- [`studio_hmi_4/stage2/`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage2)
- [`studio_hmi_4/stage3/`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage3)
- [`studio_hmi_4/sequence/`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence)
- [`studio_hmi_4/export/`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/export)
- [`studio_hmi_4/viz/`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/viz)

## 2. System flow

Let:

- $t$ be the frame index,
- $v$ be the camera/view index,
- $I_t^{(v)}$ be the input image,
- $\mathbf{u}_{t,j}^{(v)} \in \mathbb{R}^2$ be the observed 2D keypoint $j$,
- $\mathbf{X}_{t,j} \in \mathbb{R}^3$ be the triangulated 3D target point,
- $\hat{\mathbf{Y}}_{t,j} \in \mathbb{R}^3$ be the MHR forward-model keypoint.

The cleaned pipeline is:

$$
I_t^{(v)}
\rightarrow
\text{Stage1}
\rightarrow
\text{Stage2}
\rightarrow
\text{Stage3}
\rightarrow
\text{Recovery/Smoothing}
\rightarrow
\text{Export}
$$

## 3. Shared contract layer

Implemented in [`studio_hmi_4/common/contracts.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/common/contracts.py).

Validated boundaries:

- Stage-1 prediction dict,
- Stage-2 triangulation bundle,
- Stage-3 optimization result dict.

This is important because the pipeline still uses `.npy`/`.npz` artifacts between stages, so validation at boundaries is the main protection against silent drift.

Other shared helpers now live in:

- [`studio_hmi_4/common/npy_io.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/common/npy_io.py)
- [`studio_hmi_4/common/mesh_io.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/common/mesh_io.py)
- [`studio_hmi_4/common/sorting.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/common/sorting.py)
- [`studio_hmi_4/common/frame_files.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/common/frame_files.py)
- [`studio_hmi_4/common/cv2_compat.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/common/cv2_compat.py)

## 4. Stage 1: per-view inference

Public API:

- [`studio_hmi_4/stage1/runner.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage1/runner.py)

Internal split:

- [`studio_hmi_4/stage1/io.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage1/io.py)
- [`studio_hmi_4/stage1/fusion.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage1/fusion.py)

### 4.1 Base inference

For each image:

$$
\mathbf{z}_t^{(v)} = f_{\theta}(I_t^{(v)})
$$

where $\mathbf{z}_t^{(v)}$ is the SAM-3D output dict.

The runner:

1. discovers images,
2. loads the estimator,
3. selects a single person,
4. optionally applies specialized hand fusion,
5. validates and saves the final dict.

### 4.2 Person selection

If multiple detections exist, the default selection rule is largest bounding box:

$$
\hat{o} = \arg\max_o \text{area}(\text{bbox}_o)
$$

This logic is now isolated in `stage1/io.py`.

### 4.3 Specialized hand fusion

Hand fusion is now isolated in `stage1/fusion.py`.

The code keeps SAM as the base 70-keypoint layout and replaces accepted hand points through a fixed MANO/OpenPose(21) to MHR70 mapping.

For each specialized candidate $c$, the main score is wrist distance:

$$
d(c) = \left\| \mathbf{h}_{\text{wrist}}^{(c)} - \mathbf{u}_{\text{SAM,wrist}} \right\|_2
$$

subject to:

$$
d(c) < \tau_{\text{wrist}}
$$

If the SAM wrist is invalid, the fallback score becomes larger hand-box area.

### 4.4 Stage-1 outputs

Main artifact:

$$
\texttt{inference/npy/<rel\_dir>/<cam>.npy}
$$

Metadata artifact:

$$
\texttt{inference/stage1\_meta.json}
$$

The metadata file is used by the sequence orchestrator for cache reuse.

## 5. Stage 2: triangulation

Public API:

- [`studio_hmi_4/stage2/runner.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage2/runner.py)

Internal split:

- [`studio_hmi_4/stage2/io.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage2/io.py)
- [`studio_hmi_4/stage2/subset.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage2/subset.py)
- [`studio_hmi_4/stage2/camera.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage2/camera.py)
- [`studio_hmi_4/stage2/triangulation.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage2/triangulation.py)

### 5.1 Supervised subset

The triangulated subset size is:

$$
M = 20 + 20 + 6 = 46
$$

It contains:

- right hand,
- left hand,
- left/right shoulders,
- left/right elbows,
- left/right wrists.

Subset construction now has its own module, which makes the MHR name/index mapping explicit and testable.

### 5.2 Camera model

For each view $v$, the camera model loads:

- $K_v$,
- $D_v$,
- Rodrigues rotation $r_v$,
- translation $t_v$,
- image size $(w_v, h_v)$.

Projection model:

$$
\mathbf{x}_{\text{cam}}^{(v)} = R_v \mathbf{X} + t_v
$$

and reprojection:

$$
\hat{\mathbf{u}}^{(v)} = \pi_v(\mathbf{X}; K_v, D_v)
$$

### 5.3 Robust initialization

For each subset point:

1. triangulate from each camera pair,
2. score each 3D candidate by multi-view reprojection error,
3. keep the best candidate,
4. define inlier views by reprojection threshold,
5. optionally reseed from the selected inlier views.

If $e_v$ is the reprojection error in view $v$, the score is one of:

- median,
- trimmed mean,
- Huber.

### 5.4 LM refinement

The 3D point is refined by per-point LM:

$$
\min_{\mathbf{X}} \sum_v \rho\!\left(\left\|\pi_v(\mathbf{X}) - \mathbf{u}^{(v)}\right\|_2\right)
$$

where $\rho$ is either plain quadratic or Huber-weighted depending on the configuration.

### 5.5 Stage-2 outputs

Main artifact:

$$
\texttt{triangulated.npz}
$$

Important fields:

- `subset_indices`
- `subset_names`
- `points3d_init`
- `points3d_refined`
- `best_pair_idx`
- `inlier_mask`
- per-camera observed/projection/error arrays.

## 6. Stage 3: optimization

Public API:

- [`studio_hmi_4/stage3/runner.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage3/runner.py)

Internal split:

- [`studio_hmi_4/stage3/types.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage3/types.py)
- [`studio_hmi_4/stage3/runtime.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage3/runtime.py)
- [`studio_hmi_4/stage3/alignment.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage3/alignment.py)
- [`studio_hmi_4/stage3/pipeline.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage3/pipeline.py)
- [`studio_hmi_4/stage3/debug.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/stage3/debug.py)

### 6.1 View selection

Each available Stage-1 view is re-forwarded through MHR and aligned to the triangulated subset.

Similarity fit:

$$
\mathbf{Y} \approx s(\mathbf{X}R^\top) + t
$$

using weighted Umeyama alignment.

The best initialization is selected by:

1. lowest weighted 3D residual,
2. tie-break by mean reprojection error from Stage 2.

### 6.2 Optimization objective

The optimized variables are body pose and optionally hand pose.

The main data term is:

$$
\mathcal{L}_{\text{data}}
=
\frac{\sum_j w_j \, \rho\!\left(\left\|\hat{\mathbf{Y}}_{t,j}^{\text{aligned}} - \mathbf{X}_{t,j}\right\|_2\right)}
{\sum_j w_j + \epsilon}
$$

Additional terms may include:

- pose regularization,
- temporal prior,
- temporal velocity prior,
- temporal acceleration prior,
- hand regularization.

### 6.3 Masking and freeze rules

The alignment/masking logic is now isolated in `alignment.py`.

Implemented policies include:

- ignore hand dimensions in the base keep mask,
- optionally freeze lower-body dimensions,
- optional anchor-only similarity alignment,
- bad-loss classification from threshold and growth ratios.

### 6.4 Stage-3 outputs

Main artifact:

$$
\texttt{opt\_out.npy}
$$

This remains a SAM-style dict for compatibility, but now:

- contract validation is explicit,
- runtime/model loading is separated from the optimization flow,
- debug plots and `.npz` diagnostics are isolated from the math helpers.

## 7. Sequence orchestration

Public API:

- [`studio_hmi_4/sequence/runner.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence/runner.py)

Internal split:

- [`studio_hmi_4/sequence/types.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence/types.py)
- [`studio_hmi_4/sequence/discovery.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence/discovery.py)
- [`studio_hmi_4/sequence/recovery.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence/recovery.py)
- [`studio_hmi_4/sequence/orchestrator.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence/orchestrator.py)
- [`studio_hmi_4/sequence/summary.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence/summary.py)
- [`studio_hmi_4/sequence/temporal.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/sequence/temporal.py)

### 7.1 Discovery

The orchestrator discovers frame directories and available cameras under the inferred Stage-1 `npy` root.

If frame directories are simple numeric leaves, missing indices can be injected as synthetic gap entries for later recovery.

### 7.2 Cache reuse

Stage-1 reuse depends on:

- existing `inference/npy`,
- matching `stage1_meta.json`,
- enough available views in the requested frame directory.

### 7.3 Recovery

Recovery is pose-focused by design.

If a frame is missing or marked bad, the system may:

- interpolate between nearest valid neighbors,
- copy the previous valid frame,
- copy the next valid frame.

Optimization metrics are removed from recovered frames because they are no longer true measurements of that frame.

### 7.4 Smoothing

The smoothing layer works over per-frame dicts and writes:

$$
\texttt{opt\_out\_smoothed.npy}
$$

per frame.

### 7.5 Sequence summaries

The sequence layer writes:

- `pipeline_config.json`
- `sequence_summary.json`

These summarize configuration and per-frame status.

## 8. Export and visualization

### Export

Current modules:

- [`studio_hmi_4/export/compact.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/export/compact.py)
- [`studio_hmi_4/export/official_mhr.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/export/official_mhr.py)
- [`studio_hmi_4/export/common.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/export/common.py)

Shared discovery of per-frame files is now centralized instead of duplicated.

### Viewer

Current module:

- [`studio_hmi_4/viz/viewer.py`](/home/love_you/Documents/studio_hmi_4/studio_hmi_4/viz/viewer.py)

OpenCV and Open3D are now runtime-checked instead of import-time hard requirements, which improves portability when the viewer is not being used.

## 9. Copy-into-`sam-3d-body` compatibility

The cleaned codebase was explicitly checked for the following deployment pattern:

1. copy the repo files/directories into the first-level root of a `sam-3d-body` repository;
2. run imports and synthetic tests from that new root.

Verified result:

- root wrappers imported successfully from the copied root,
- synthetic tests passed from the copied root.

That works because:

- root wrappers keep the old command names,
- the package `studio_hmi_4/` is self-contained,
- the stage runtimes still expect host-repo modules like `sam_3d_body` and `tools.*`,
- `mhr70.py` stays at the root where existing code expects it.

Detailed instructions are in [`docs/integration_guide.md`](/home/love_you/Documents/studio_hmi_4/docs/integration_guide.md).

## 10. Verification status

Verified locally:

- compile pass for package modules and wrappers,
- synthetic shape/integration tests,
- copied-root wrapper import test,
- copied-root synthetic test run.

Not verified here:

- real-data numerical regression against a frozen baseline,
- long-running production sequences,
- exact behavior against every external dependency version.

That limitation exists because Phase 1 baseline capture was intentionally skipped.
