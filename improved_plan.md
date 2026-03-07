# improved_plan

## 1. Goal

Clean the repository without breaking the current pipeline behavior, then make the algorithms easier to understand, easier to tune, and easier to extend.

This plan is intentionally behavior-preserving first, algorithm-improving second.

## 2. Main findings that drive the cleanup

### 2.1 Monolithic scripts

The four core files are doing too much:

- `sam3d_inference.py`
- `triangulate_mhr3d_gt.py`
- `optimize_mhr_pose.py`
- `run_full_pipeline.py`

Each mixes configuration, file I/O, math, orchestration, debug output, and serialization.

### 2.2 Repeated utilities

The same logic appears in multiple files:

- `.npy` dict loading,
- natural sorting,
- PLY writing,
- frame discovery,
- path normalization.

This increases maintenance cost and the chance of subtle behavior drift.

### 2.3 Implicit stage contracts

The pipeline works by passing large pickled dicts between stages, but there is no formal schema layer.

That means:

- hidden assumptions are everywhere,
- refactoring is risky,
- debugging requires reading code instead of reading contracts.

### 2.4 Configuration sprawl

Important defaults are spread across:

- CLI definitions,
- dataclass defaults,
- `namespace_to_config`,
- helper functions.

This makes the real runtime configuration harder to understand than it should be.

### 2.5 Repo-root clutter

The root currently mixes:

- executable scripts,
- internal docs,
- planning notes,
- generated `.pdf/.log/.aux/.out` artifacts.

That makes the repo feel less like a maintained codebase and more like a working scratch directory.

### 2.6 Missing verification layer

There are no dedicated tests for:

- stage contracts,
- triangulation math,
- optimization masking,
- sequence recovery,
- export integrity.

That is the main blocker to safe cleanup.

## 3. Refactor principles

1. Preserve outputs first.
2. Move shared logic before changing math.
3. Formalize contracts before shrinking CLI.
4. Add tests before algorithmic rewrites.
5. Separate "public user interface" from "internal tuning knobs".

## 4. Phase-by-phase plan

## Phase 1: Freeze current behavior

### Purpose

Capture the current pipeline as a known baseline.

### Work

- Document exact stage inputs and outputs.
- Save one or two reference runs and keep their summaries.
- Record representative commands for:
  - full pipeline,
  - hand-precompute + full pipeline,
  - export,
  - viewer.
- Create a minimal benchmark set of frames and one short sequence.

### Deliverables

- behavior baseline folder,
- reference `sequence_summary.json`,
- contract notes per stage,
- benchmark input set.

### Acceptance

- We can rerun the current pipeline and compare new outputs to known baseline artifacts.

## Phase 2: Create a real module structure

### Purpose

Move from root-level giant scripts to a small internal package while keeping CLI entrypoints thin.

### Work

Create a structure like:

```text
src/studio_hmi_4/
  common/
  stage1/
  stage2/
  stage3/
  sequence/
  export/
  viz/
```

Keep top-level scripts as wrappers that only:

- parse args,
- call package functions,
- print summaries.

### Deliverables

- internal package layout,
- thin CLI wrappers,
- imports updated to package modules.

### Acceptance

- command-line behavior remains unchanged,
- file paths and outputs remain unchanged.

## Phase 3: Centralize schemas and I/O

### Purpose

Make stage contracts explicit.

### Work

- Define typed structures for:
  - Stage-1 frame output,
  - Stage-2 triangulation result,
  - Stage-3 optimization result,
  - sequence summary entry.
- Create shared loaders/savers for:
  - `.npy` dict payloads,
  - `.npz` bundles,
  - summary JSON,
  - PLY writing.
- Add one shared path and sorting utility module.

### Deliverables

- `schemas.py` or equivalent typed contract layer,
- `io.py` and `mesh_io.py`,
- one natural-sort helper,
- one shared frame-discovery helper.

### Acceptance

- duplicated helper code disappears from stage scripts,
- stage outputs can be validated at load time.

## Phase 4: Simplify configuration

### Purpose

Separate essential user inputs from advanced tuning knobs.

### Work

- Keep a small public CLI for `run_full_pipeline.py`.
- Move advanced parameters into:
  - presets,
  - config files,
  - internal constants.
- Introduce named presets, for example:
  - `default`
  - `hand_focus`
  - `debug_single_frame`
  - `robust_sparse_views`
- Ensure one place defines default values.

### Deliverables

- central config model,
- preset definitions,
- reduced public CLI,
- config dump written into output root for each run.

### Acceptance

- a user can understand the normal pipeline from one help screen,
- advanced settings are still available but not cluttering the main interface.

## Phase 5: Split algorithmic responsibilities by stage

### Stage 1 refactor target

Split into:

- model loading,
- image collection,
- person selection,
- hand fusion,
- persistence/debug output.

Key benefit:

- hand-fusion logic becomes independently testable,
- Stage 1 becomes understandable without reading file I/O and mesh debug code at the same time.

### Stage 2 refactor target

Split into:

- subset builder,
- camera rig loader,
- DLT initializer,
- robust scorer,
- LM refiner,
- artifact writer.

Key benefit:

- triangulation math becomes readable and benchmarkable.

### Stage 3 refactor target

Split into:

- runtime/model head construction,
- view scoring,
- alignment utilities,
- optimization objective,
- bad-frame classifier,
- output writer.

Key benefit:

- optimization logic can be tested without touching file-system code.

### Sequence refactor target

Split `run_full_pipeline.py` into:

- stage orchestration,
- cache reuse logic,
- recovery policy,
- smoothing policy,
- summary generation.

Key benefit:

- sequence control flow becomes much easier to reason about.

## Phase 6: Add verification and regression tests

### Purpose

Make cleanup safe.

### Work

Add tests for:

- Stage 1:
  - person selection,
  - hand mapping,
  - wrist-distance gating.
- Stage 2:
  - DLT triangulation,
  - robust score modes,
  - inlier gating,
  - LM refinement stability.
- Stage 3:
  - valid-point masking,
  - lower-body freeze mask,
  - bad-loss classification,
  - temporal initialization.
- Sequence:
  - gap injection,
  - interpolation recovery,
  - copy recovery,
  - smoothing.
- Export:
  - compact parameter dimensions,
  - mesh regeneration path,
  - bad-frame skipping behavior.

### Deliverables

- unit tests,
- small integration tests,
- one smoke test that runs a tiny end-to-end pipeline fixture.

### Acceptance

- cleanup changes can be validated automatically before reuse.

## Phase 7: Improve algorithm clarity and robustness

This phase should begin only after Phases 1-6 are complete.

### Recommended algorithm improvements

1. Make Stage-1 hand fusion confidence-aware rather than wrist-distance-only.
2. Add explicit confidence propagation from Stage 1 into Stage 2 and Stage 3.
3. Replace implicit "weight from inlier-mask mean" with a clearer confidence model.
4. Separate alignment points from supervision points in a more explicit config object.
5. Introduce experiment logging for parameter sweeps and quality metrics.

### Not recommended before cleanup

- rewriting the math stack,
- changing core file formats,
- changing objective terms aggressively,
- collapsing all stages into one end-to-end block.

## 5. File-specific recommended end state

### `sam3d_inference.py`

- keep as CLI wrapper only;
- move logic into `stage1/runner.py`, `stage1/fusion.py`, `stage1/io.py`.

### `triangulate_mhr3d_gt.py`

- keep as CLI wrapper only;
- move logic into `stage2/camera.py`, `stage2/subset.py`, `stage2/triangulation.py`, `stage2/io.py`.

### `optimize_mhr_pose.py`

- keep as CLI wrapper only;
- move logic into `stage3/runtime.py`, `stage3/objective.py`, `stage3/alignment.py`, `stage3/io.py`.

### `run_full_pipeline.py`

- keep as CLI wrapper only;
- move logic into `sequence/orchestrator.py`, `sequence/recovery.py`, `sequence/smoothing.py`, `sequence/summary.py`.

### `video_temporal_utils.py`

- fold into `sequence/` and stop duplicating I/O helpers elsewhere.

### `export_mhr_final_results.py` and `run_mhr_repo_from_export.py`

- move shared export helpers into `export/common.py`.

### `interactive_mesh_video_viewer.py`

- keep as standalone tool, but move sequence discovery and `.npy` loading into shared code.

## 6. Documentation cleanup plan

### Work

- Move stable docs into `docs/`.
- Move historical notes and paper drafts into `docs/archive/`.
- Remove generated `.pdf/.log/.aux/.out` from the root or move them to `docs/generated/`.
- Keep only one current architecture doc, one parameter doc, and one usage guide as the active source of truth.

### Acceptance

- a new developer can understand the repo by reading three files, not twelve overlapping notes.

## 7. Suggested execution order

1. baseline and test fixtures,
2. package structure,
3. shared schemas and I/O,
4. CLI simplification,
5. stage-by-stage code split,
6. tests,
7. algorithm refinement,
8. doc/archive cleanup.

That order reduces the chance of mixing structural cleanup with mathematical changes.

## 8. Success criteria

The cleanup should be considered successful when:

- the pipeline produces equivalent or better outputs on the baseline set,
- the main runner has a small understandable CLI,
- each stage has an explicit data contract,
- duplicated helpers are removed,
- the repo root is no longer cluttered,
- basic regression tests exist,
- future algorithm changes can be made without editing giant scripts.
