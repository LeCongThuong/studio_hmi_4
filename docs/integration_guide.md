# Integration Guide

## Goal

This guide explains how to use this cleaned pipeline codebase in two ways:

1. run it as its own repo;
2. copy it into the first-level root of a `sam-3d-body` repository.

The second mode is supported because the stage wrappers stay at the root and the implementation lives in the `studio_hmi_4/` package.

## What to copy into `sam-3d-body`

Copy these files/directories into the root of the target `sam-3d-body` repo:

- `studio_hmi_4/`
- `tests/` if you want the synthetic verification suite
- `mhr70.py`
- `sam3d_inference.py`
- `triangulate_mhr3d_gt.py`
- `optimize_mhr_pose.py`
- `run_full_pipeline.py`
- `video_temporal_utils.py`
- `export_mhr_final_results.py`
- `run_mhr_repo_from_export.py`
- `run_wilor_precompute.py`
- `extract_per_tick_frames_from_csv.py`
- `interactive_mesh_video_viewer.py`

You can also copy the docs:

- `README.md`
- `course_detail.md`
- `details.md`
- `detail_after.md`
- `params.md`
- `improved_plan.md`
- `docs/`

## Expected target layout

After copying, the target repo root should look like:

```text
sam-3d-body/
  sam_3d_body/
  tools/
  studio_hmi_4/
  mhr70.py
  sam3d_inference.py
  triangulate_mhr3d_gt.py
  optimize_mhr_pose.py
  run_full_pipeline.py
  export_mhr_final_results.py
  run_mhr_repo_from_export.py
  run_wilor_precompute.py
  extract_per_tick_frames_from_csv.py
  interactive_mesh_video_viewer.py
```

This works because:

- root scripts import from `studio_hmi_4.*`;
- stage-1 runtime imports still resolve `sam_3d_body` and `tools.*` from the target repo root;
- `mhr70.py` remains available at the root for subset metadata.

## Verified compatibility

The copied-root workflow was verified by:

1. copying the package directory, wrappers, `mhr70.py`, and tests into a temporary fake root;
2. importing the root wrappers from that copied root;
3. running `python3 -m unittest -v tests.test_synthetic_shapes` from that copied root.

The verification passed.

## Dependency behavior

Some dependencies are optional at import time now:

- `cv2` is loaded lazily and raises only when a feature that truly needs OpenCV is executed;
- `open3d` is loaded lazily for the interactive viewer only;
- `sam_3d_body` remains required for real stage-1/stage-3 runtime execution;
- the synthetic tests do not require real capture data.

That means copying the files into another repo root no longer breaks simple imports when viewer/OpenCV dependencies are absent.

## Recommended verification after copying

From the target `sam-3d-body` root:

```bash
python3 -m py_compile $(find studio_hmi_4 -name '*.py' | sort) \
  sam3d_inference.py triangulate_mhr3d_gt.py optimize_mhr_pose.py \
  run_full_pipeline.py video_temporal_utils.py export_mhr_final_results.py \
  run_mhr_repo_from_export.py run_wilor_precompute.py \
  extract_per_tick_frames_from_csv.py interactive_mesh_video_viewer.py
```

If you copied the tests too:

```bash
python3 -m unittest -v tests.test_synthetic_shapes
```

## Operational note

The synthetic tests only verify interface/shape behavior and orchestration wiring. They do not replace a real-data regression run inside your actual `sam-3d-body` environment.
