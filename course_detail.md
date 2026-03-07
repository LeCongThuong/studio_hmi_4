# course_detail

## 1. Repository purpose

This repository is now a cleaned, package-backed multi-view human reconstruction pipeline built around SAM-3D-Body and MHR.

Its real workflow is:

1. prepare synchronized multi-camera frames,
2. run per-view SAM inference,
3. triangulate a robust 3D supervision subset,
4. optimize MHR pose parameters against that subset,
5. recover and smooth sequence failures,
6. export meshes and compact MHR-ready parameters,
7. inspect outputs with viewer/debug tools.

The old giant-script structure has been replaced with thin root entrypoints plus an internal package.

## 2. Current architecture

The root scripts are compatibility wrappers only:

- `sam3d_inference.py`
- `triangulate_mhr3d_gt.py`
- `optimize_mhr_pose.py`
- `run_full_pipeline.py`
- `export_mhr_final_results.py`
- `run_mhr_repo_from_export.py`
- `run_wilor_precompute.py`
- `extract_per_tick_frames_from_csv.py`
- `interactive_mesh_video_viewer.py`

The actual implementation lives in `studio_hmi_4/`.

## 3. Package map

### `studio_hmi_4/common/`

Shared code:

- stage contracts,
- `.npy` dict I/O,
- mesh I/O,
- sorting helpers,
- shared frame discovery,
- optional OpenCV compatibility layer.

### `studio_hmi_4/stage1/`

Stage-1 SAM inference:

- `runner.py`: public API and CLI-facing behavior,
- `io.py`: file discovery and persistence,
- `fusion.py`: specialized hand fusion logic.

### `studio_hmi_4/stage2/`

Stage-2 triangulation:

- `runner.py`: public API and orchestration,
- `io.py`: input loading and debug overlays,
- `subset.py`: MHR subset building,
- `camera.py`: calibrated camera model and TOML parsing,
- `triangulation.py`: DLT, robust scoring, LM refinement.

### `studio_hmi_4/stage3/`

Stage-3 optimization:

- `runner.py`: public API and CLI-facing behavior,
- `types.py`: optimization dataclasses,
- `runtime.py`: SAM/MHR runtime loading and forward helpers,
- `alignment.py`: similarity, masking, and bad-loss helpers,
- `pipeline.py`: core optimization procedure,
- `debug.py`: plots and diagnostics.

### `studio_hmi_4/sequence/`

Sequence orchestration:

- `runner.py`: public API and patch point for stage hooks,
- `types.py`: frame/sequence dataclasses,
- `discovery.py`: frame discovery and metadata expectations,
- `recovery.py`: recovery logic and temporal-state helpers,
- `orchestrator.py`: full pipeline control flow,
- `summary.py`: summary/config writing,
- `temporal.py`: smoothing and sequence-level output helpers.

### `studio_hmi_4/export/`

Export and replay:

- `compact.py`: export compact MHR-ready per-frame outputs,
- `official_mhr.py`: replay official MHR forward on exported params,
- `common.py`: shared export discovery and parameter extraction helpers.

### `studio_hmi_4/viz/`

Viewer tooling:

- `viewer.py`: interactive mesh playback with optional front-view media.

### `studio_hmi_4/tools/`

Utility scripts:

- `extract_frames.py`
- `wilor_precompute.py`

## 4. Pipeline stages

### Stage 0: optional frame extraction

`extract_per_tick_frames_from_csv.py`

- reads recorder metadata and videos,
- writes `OUT/<frame_id>/<cam>.jpg`,
- creates frame manifests.

### Stage 0b: optional specialized-hand precompute

`run_wilor_precompute.py`

- runs WiLoR-mini separately,
- stores hand detections for later fusion in Stage 1.

### Stage 1: per-view inference

`sam3d_inference.py` -> `studio_hmi_4/stage1/`

- runs SAM-3D-Body on each image,
- selects one person,
- optionally fuses specialized hand keypoints into the 70-point MHR layout,
- saves per-image `.npy` dicts and optional side outputs.

### Stage 2: triangulation and refinement

`triangulate_mhr3d_gt.py` -> `studio_hmi_4/stage2/`

- loads Stage-1 2D observations,
- builds the 46-point subset,
- loads calibrated cameras,
- performs robust pairwise DLT initialization,
- gates inlier views,
- runs per-point LM refinement,
- saves a rich `.npz` triangulation bundle.

### Stage 3: MHR pose optimization

`optimize_mhr_pose.py` -> `studio_hmi_4/stage3/`

- scores each view-specific SAM initialization,
- picks the best initialization,
- optimizes body pose and optionally hand pose,
- supports temporal priors, fixed non-pose parameters, and lower-body freeze,
- saves optimized SAM-style `.npy` dicts with `opt_*` diagnostics.

### Sequence orchestration

`run_full_pipeline.py` -> `studio_hmi_4/sequence/`

- runs Stage 1, 2, and 3 over one frame or many frames,
- reuses Stage-1 cache when metadata matches,
- recovers missing/bad frames,
- smooths valid outputs,
- writes `pipeline_config.json` and `sequence_summary.json`,
- optionally saves a sequence MP4.

### Export and viewing

- `export_mhr_final_results.py`: compact export for downstream MHR use,
- `run_mhr_repo_from_export.py`: official MHR replay,
- `interactive_mesh_video_viewer.py`: interactive playback.

## 5. Main data contracts

### Stage 1 output

One `.npy` per image with keys such as:

- `pred_keypoints_2d`
- `body_pose_params`
- `hand_pose_params`
- `scale_params`
- `shape_params`
- `expr_params`
- `pred_vertices`
- `pred_keypoints_3d`

Validated by `studio_hmi_4/common/contracts.py`.

### Stage 2 output

One `.npz` bundle with fields such as:

- `points3d_refined`
- `subset_indices`
- `subset_names`
- `inlier_mask`
- per-camera projections and reprojection errors.

### Stage 3 output

One `.npy` dict with:

- optimized MHR parameters,
- aligned keypoints/vertices/joints,
- similarity transform,
- loss curves and quality flags.

### Sequence output

- per-frame `opt_out.npy`,
- per-frame `opt_out_smoothed.npy`,
- `sequence_summary.json`,
- `pipeline_config.json`.

## 6. What was improved

The cleanup now addresses the original main findings:

- monolithic stage scripts were split into smaller modules,
- repeated utilities were centralized,
- contracts are explicit and validated,
- full-pipeline defaults are centralized,
- export/viewer discovery is shared,
- synthetic regression tests exist,
- root clutter was reduced by moving archive/generated docs under `docs/`.

## 7. Integration into `sam-3d-body`

This codebase can be copied into the first-level root of a `sam-3d-body` repository.

Why it still works:

- root wrappers stay at the root,
- the `studio_hmi_4/` package stays self-contained,
- stage runtime imports still resolve `sam_3d_body` and `tools.*` from the host repo,
- `mhr70.py` remains available at the root.

The verified copy workflow is documented in `docs/integration_guide.md`.

## 8. Practical summary

The repo should now be understood as:

1. stable public root commands,
2. internal package modules by stage/responsibility,
3. explicit file contracts between stages,
4. synthetic tests for shape and orchestration safety,
5. compatibility with embedding into a `sam-3d-body` root.
