# detail_after

## 1. Actual repository state after cleanup

The repository is no longer a set of giant top-level implementation scripts.

It is now:

- root wrapper scripts for compatibility,
- an internal `studio_hmi_4/` package with per-stage submodules,
- centralized contracts and utilities,
- synthetic tests,
- cleaned docs under `docs/`.

## 2. Actual code tree

```text
studio_hmi_4/
  common/
    contracts.py
    cv2_compat.py
    frame_files.py
    mesh_io.py
    npy_io.py
    sorting.py
  stage1/
    runner.py
    io.py
    fusion.py
  stage2/
    runner.py
    io.py
    subset.py
    camera.py
    triangulation.py
  stage3/
    runner.py
    types.py
    runtime.py
    alignment.py
    pipeline.py
    debug.py
  sequence/
    runner.py
    types.py
    discovery.py
    recovery.py
    orchestrator.py
    summary.py
    temporal.py
    presets.py
  export/
    compact.py
    official_mhr.py
    common.py
  tools/
    extract_frames.py
    wilor_precompute.py
  viz/
    viewer.py
```

## 3. What the root scripts do now

The root files remain:

- `sam3d_inference.py`
- `triangulate_mhr3d_gt.py`
- `optimize_mhr_pose.py`
- `run_full_pipeline.py`
- `export_mhr_final_results.py`
- `run_mhr_repo_from_export.py`
- `run_wilor_precompute.py`
- `extract_per_tick_frames_from_csv.py`
- `interactive_mesh_video_viewer.py`

But they now act as compatibility wrappers so existing command usage does not need to change.

## 4. Actual contract state

The pipeline now has explicit validation for:

- Stage-1 prediction dicts,
- Stage-2 triangulation bundles,
- Stage-3 optimization result dicts.

These validations live in `studio_hmi_4/common/contracts.py` and are used at save/load boundaries.

## 5. Actual test state

The repository now includes shape-focused synthetic verification in:

- `tests/test_synthetic_shapes.py`

Covered behaviors:

- Stage-1 output shapes,
- Stage-2 triangulation bundle shapes,
- Stage-3 optimization output shapes,
- sequence orchestration/recovery wiring,
- compact export dimensions.

These tests use synthetic/random data and fake runtimes, not real captures.

## 6. Actual integration state with `sam-3d-body`

The codebase was checked for the copy-into-host-repo scenario.

Verified behavior:

1. copy the package, wrappers, `mhr70.py`, and tests into a temporary first-level root,
2. import the root wrappers from that copied root,
3. run the synthetic test suite from that copied root.

Result:

- wrapper imports succeeded,
- the synthetic test suite passed.

This means the cleaned codebase is compatible with being placed at the first-level depth of a `sam-3d-body` repository, assuming normal runtime dependencies are present when real stages are executed.

## 7. Import robustness improvements

The cleanup also fixed import-time fragility:

- OpenCV is now optional at import time and only required when OpenCV-backed features are actually used,
- Open3D is now optional at import time and only required when the viewer is executed.

That prevents unrelated commands from failing just because viewer/debug dependencies are absent.

## 8. Docs state after cleanup

The active docs now should be read in this order:

1. `README.md`
2. `course_detail.md`
3. `details.md`
4. `docs/integration_guide.md`
5. `params.md`

Historical notes and generated paper artifacts were moved under `docs/archive/` and `docs/generated/`.

## 9. Remaining practical limitation

The cleanup was verified with synthetic shape tests and copied-root checks, but not with a preserved real-data baseline run because Phase 1 baseline capture was skipped.

So:

- structural compatibility is verified,
- contract correctness is verified,
- orchestration wiring is verified,
- real-scene numerical equivalence is not yet baseline-compared.
