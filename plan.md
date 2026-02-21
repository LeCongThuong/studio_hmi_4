# Implementation Plan: Fix Pipeline Bugs + Add Lower-Body Freeze

## 1) Objective

Implement code changes that:

1. Fix all confirmed pipeline bugs from `research.md`.
2. Add a feature to lock lower-body motion so those parts are not optimized (to reduce jitter/drift when the person is effectively stationary).
3. Add runtime guardrails and clear failure modes so regressions are detected quickly during pipeline runs.

This document is implementation-focused and includes concrete code snippets.

---

## 2) Scope

Target files:

- `optimize_mhr_pose.py`
- `run_full_pipeline.py`
- `video_temporal_utils.py`
- `triangulate_mhr3d_gt.py`
- `sam3d_inference.py`

---

## 3) Bug-to-Fix Map

## B1 + B5: NaN/Inf GT points and stage contract mismatch

Fix in Stage-3 optimization:

- Build a robust valid-point mask before Umeyama and loss:
  - point must be finite in triangulated GT.
  - point must be finite in predicted subset.
- Require minimum valid points before optimization; otherwise return/mark frame as unusable.

## B2: All-zero inlier weights produce false-perfect losses

Fix in Stage-3 optimization:

- If weight sum is near zero:
  - fallback to uniform weights over finite points (default), or
  - fail frame if strict mode enabled.

## B3: stale `final_loss` used in bad-frame classification

Fix in Stage-3 optimization:

- Recompute final objective after restoring best pose.
- Classify bad frame using that recomputed final objective.

## B4: smoothing propagates NaNs

Fix in sequence smoothing:

- Treat frame values as valid only if all finite.
- Exclude non-finite frames from smoothing anchors.
- Ensure nearest-fill pulls from finite anchors only.

## Issue-A: first-person-only output selection in Stage-1

Fix in Stage-1 inference:

- Add configurable selection strategy:
  - `first`
  - `largest_bbox` (recommended default)
  - optional fixed `person_index`

## Issue-B: `.npy` plain-array loading brittle

Fix in Stage-2 loader:

- Support direct ndarray payload (e.g. `(70,2)` or `(N,70,2)`) without `.item()`.

## Issue-C: stale Stage-1 reuse in `run_full_pipeline`

Fix in orchestrator:

- Save stage-1 metadata (input root, frame_rel, model identifiers).
- Reuse stage-1 only if metadata matches current request.

## Issue-D: camera file wildcard ambiguity

Per your note, this is intentionally out of scope for now and will not be changed in this implementation pass.

---

## 4) New Feature: Lower-Body Freeze (No Optimization on Lower Part)

## Feature behavior

When enabled:

- lower-body pose dimensions stay fixed to initialization.
- optimizer updates only upper-body-relevant pose dimensions.

This is a hard lock, not just smoothing.

## Recommended interface

Per your note, the lower-body feature exposes only one parameter:

- `--freeze_lower_body` (bool)

In `run_full_pipeline.py`, pass through only:

- `--freeze_lower_body`

## How to obtain lower-body indices (hardcoded from one-time inspection)

Per your note, this will be hardcoded, not discovered at runtime.

Plan:

1. Inspect `mhr70.py` and upstream SAM-3D/MHR mapping code once to identify:
   - lower-body keypoints (hips, knees, ankles, toes, heels)
   - corresponding body-pose parameter indices that drive those keypoints
2. Hardcode lower-body pose index tables in code.
3. Support different pose-vector sizes by table key (for example, 133 and 204 if present in your runtime/checkpoint path).
4. At runtime, when `--freeze_lower_body` is set, pick the table by `pose_dim` and freeze those dimensions only.

This keeps the user interface minimal (`--freeze_lower_body` only) and avoids unstable runtime sensitivity estimation.

---

## 5) Detailed Implementation Steps

## Step 1: Harden Stage-3 config and helpers

Edit `OptimizationConfig`:

```python
@dataclass
class OptimizationConfig:
    # existing fields...
    min_valid_points: int = 6
    zero_weight_strategy: str = "uniform_finite"  # uniform_finite | fail

    freeze_lower_body: bool = False
```

Add helpers in `optimize_mhr_pose.py`:

```python
def _sanitize_subset_and_weights(gtM: np.ndarray, wM: np.ndarray, min_valid_points: int, strategy: str):
    finite = np.isfinite(gtM).all(axis=1)
    w = np.asarray(wM, dtype=np.float32).reshape(-1)
    w[~finite] = 0.0

    nonzero = (w > 1e-8) & finite
    if int(nonzero.sum()) < int(min_valid_points):
        if strategy == "uniform_finite":
            w = np.where(finite, 1.0, 0.0).astype(np.float32)
            nonzero = finite
        else:
            raise RuntimeError(
                f"Insufficient valid weighted points: {int(nonzero.sum())} < {int(min_valid_points)}"
            )

    return finite, w, nonzero
```

---

## Step 2: Fix Stage-3 scoring + optimization masking

Use filtered indices for Umeyama/loss everywhere:

```python
# after loading gtM, wM
finite_mask_np, wM_np, valid_mask_np = _sanitize_subset_and_weights(
    gtM=gtM,
    wM=wM,
    min_valid_points=config.min_valid_points,
    strategy=config.zero_weight_strategy,
)
valid_idx_t = torch.from_numpy(np.where(valid_mask_np)[0]).to(device)

# torch tensors
wM_t = to_torch(wM_np, device)
gtM_t = to_torch(gtM, device)

# in per-view init scoring
predM = k70[subset_idx]
predM_v = predM.index_select(0, valid_idx_t)
gtM_v = gtM_t.index_select(0, valid_idx_t)
wM_v = wM_t.index_select(0, valid_idx_t)

if not torch.isfinite(predM_v).all():
    view_scores_3d[cam] = float("inf")
    continue

s, R, t = umeyama_similarity(predM_v, gtM_v, w=wM_v, with_scale=config.with_scale)
```

In main loop:

```python
predM = k70[subset_idx]
predM_v = predM.index_select(0, valid_idx_t)
gtM_v = gtM_t.index_select(0, valid_idx_t)
wM_v = wM_t.index_select(0, valid_idx_t)

if not torch.isfinite(predM_v).all():
    raise RuntimeError("Non-finite prediction in valid subset during optimization")

with torch.no_grad():
    s, R, t = umeyama_similarity(predM_v.detach(), gtM_v, w=wM_v, with_scale=config.with_scale)

predM_aligned_v = s * (predM_v @ R.T) + t[None, :]
r = torch.sqrt(((predM_aligned_v - gtM_v) ** 2).sum(dim=1) + 1e-12)
loss_data = (wM_v * huber(r, delta=config.huber_m)).sum() / (wM_v.sum() + 1e-9)
```

---

## Step 3: Fix stale `final_loss` classification

After restoring best pose, recompute final objective before classification:

```python
# after pose.copy_(best_pose)
with torch.no_grad():
    pose_eff = pose.detach() * optimize_mask
    outF = mhr_fk(...)
    k70 = apply_repo_camera_flip_xyz(outF[1].squeeze(0)[:70])
    predM = k70[subset_idx]

    predM_v = predM.index_select(0, valid_idx_t)
    gtM_v = gtM_t.index_select(0, valid_idx_t)
    wM_v = wM_t.index_select(0, valid_idx_t)

    s, R, t = umeyama_similarity(predM_v, gtM_v, w=wM_v, with_scale=config.with_scale)
    predM_aligned_v = s * (predM_v @ R.T) + t[None, :]
    r_v = torch.sqrt(((predM_aligned_v - gtM_v) ** 2).sum(dim=1) + 1e-12)

    final_data_loss = float(((wM_v * huber(r_v, delta=config.huber_m)).sum() / (wM_v.sum() + 1e-9)).cpu().item())
    final_reg_loss = float((config.w_pose_reg * torch.mean((pose - init_pose) ** 2)).cpu().item())
    final_loss = final_data_loss + final_reg_loss
```

Use this recomputed `final_loss`/`final_data_loss` in `classify_bad_optimization`.

---

## Step 4: Add lower-body hard freeze in Stage-3

Build optimization mask:

```python
optimize_mask = keep_mask.clone()  # existing hand+jaw freeze

if config.freeze_lower_body:
    lower_idxs = resolve_lower_body_pose_indices(int(init_pose.numel()))
    optimize_mask[lower_idxs] = 0.0

# initialize train var
a = init_pose.clone().detach()
pose = a.requires_grad_(True)
```

Enforce frozen dimensions each iteration:

```python
pose_eff = pose * optimize_mask + init_pose * (1.0 - optimize_mask)

loss.backward()
with torch.no_grad():
    if pose.grad is not None:
        pose.grad[optimize_mask == 0] = 0.0
opt.step()

with torch.no_grad():
    pose = pose * optimize_mask + init_pose * (1.0 - optimize_mask)
```

This guarantees lower-body dimensions never change from initialization.

---

## Step 5: Implement lower-body index resolution

Implement hardcoded lower-body index tables from one-time MHR inspection.

First, define lower-body keypoint names and indices from `mhr70.py`:

```python
from mhr70 import pose_info

LOWER_KPT_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

def _name_to_idx_map(pose_info_dict):
    out = {}
    for k, v in pose_info_dict["keypoint_info"].items():
        out[str(v["name"])] = int(k)
    return out

NAME_TO_IDX = _name_to_idx_map(pose_info)
LOWER_KPT_IDXS = [NAME_TO_IDX[n] for n in LOWER_KPT_NAMES]
```

Then hardcode pose-parameter freeze tables by pose-vector dimension:

```python
LOWER_BODY_POSE_IDXS_BY_DIM = {
    # Filled from one-time mapping audit:
    # 133: np.array([...], dtype=np.int64),
    # 204: np.array([...], dtype=np.int64),
}

def resolve_lower_body_pose_indices(pose_dim: int) -> np.ndarray:
    if pose_dim not in LOWER_BODY_POSE_IDXS_BY_DIM:
        raise RuntimeError(
            f"freeze_lower_body enabled but no hardcoded lower-body index table for pose_dim={pose_dim}"
        )
    return LOWER_BODY_POSE_IDXS_BY_DIM[pose_dim]
```

Implementation notes:

- Add a short docstring near the table citing source files used for the mapping (`mhr70.py`, upstream MHR/SAM pose mapping code).
- Keep freeze table maintenance explicit and versioned (if model changes, update table).
- Intersect hardcoded lower-body indices with current `keep_mask` before applying to avoid overriding existing frozen dims.

---

## Step 6: Make smoothing finite-safe

Edit `video_temporal_utils.py` inside `smooth_frame_dict_sequence`:

```python
arr = np.asarray(d[key], dtype=np.float32)
if arr.shape != sample_shape or not np.issubdtype(arr.dtype, np.number):
    continue
if not np.isfinite(arr).all():
    continue  # do not treat NaN-containing frame as valid anchor

seq[i] = arr
valid[i] = True
```

Optional extra guard before writing smoothed values:

```python
if not np.isfinite(smoothed[i]).all():
    continue
```

---

## Step 7: Fix `.npy` array loading path in triangulation

Edit `triangulate_mhr3d_gt.py` in `load_pred_keypoints_2d`:

```python
if suffix == ".npy":
    obj = np.load(str(file_path), allow_pickle=True)
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.shape == () and hasattr(obj, "item"):
            data = obj.item()
        else:
            data = {"pred_keypoints_2d": obj}
    elif isinstance(obj, dict):
        data = obj
    else:
        raise ValueError(f"Unsupported .npy payload type: {type(obj)}")
```

---

## Step 8: Improve Stage-1 person selection

Edit `sam3d_inference.py`:

- add config fields:
  - `person_select_strategy: str = "largest_bbox"`
  - `person_index: int = 0`

- change selector:

```python
def extract_primary_output(outputs, strategy="largest_bbox", person_index=0):
    if outputs is None:
        return None
    if isinstance(outputs, Mapping):
        return dict(outputs)
    if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
        return None

    cands = [o for o in outputs if isinstance(o, Mapping)]
    if not cands:
        return None

    if strategy == "first":
        return dict(cands[0])
    if strategy == "person_index":
        idx = int(np.clip(person_index, 0, len(cands)-1))
        return dict(cands[idx])

    # largest_bbox (default)
    def area(c):
        b = np.asarray(c.get("bbox", [0, 0, 0, 0]), dtype=np.float32).reshape(-1)
        if b.size < 4:
            return -1.0
        return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))

    return dict(max(cands, key=area))
```

---

## Step 9: Add Stage-1 reuse metadata check

Add metadata write in `sam3d_inference.py` after run:

```python
meta = {
    "image_folder": str(Path(config.image_folder).resolve()),
    "include_rel_dirs": config.include_rel_dirs,
    "checkpoint_path": config.checkpoint_path,
    "detector_name": config.detector_name,
    "segmentor_name": config.segmentor_name,
    "fov_name": config.fov_name,
}
(output_root / "stage1_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
```

In `run_full_pipeline.py`, reuse only if metadata matches current invocation.

---

## 6) CLI and Dataclass Wiring

## `optimize_mhr_pose.py`

Add args:

```python
ap.add_argument("--min_valid_points", type=int, default=6)
ap.add_argument("--zero_weight_strategy", choices=["uniform_finite", "fail"], default="uniform_finite")

ap.add_argument("--freeze_lower_body", action="store_true")
```

## `run_full_pipeline.py`

Pass-through args to `OptimizationConfig`:

```python
min_valid_points=config.min_valid_points,
zero_weight_strategy=config.zero_weight_strategy,
freeze_lower_body=config.freeze_lower_body,
```

---

## 7) Validation Plan

Validation in this pass is run-based (no new `tests/` files).

## Targeted checks

1. NaN GT handling:
- input NPZ with one NaN point should not crash.
- optimizer should either skip bad points and run, or return a controlled error if below `min_valid_points`.

2. Zero-weight mask case:
- all-zero `inlier_mask` should not return fake-zero-loss unless explicitly allowed.

3. Final-loss classification consistency:
- if best pose restored, classification should reflect post-restore final loss.

4. Smoothing finite safety:
- `[finite, NaN, finite]` sequence should remain finite after smoothing.

5. Lower-body freeze:
- with freeze enabled, selected lower-body `body_pose_params` indices must be unchanged from init.

## Integration checks

1. Single-frame pipeline run with known good sample.
2. Multi-frame sequence with missing views and occlusions.
3. Compare `opt_out.npy` vs `opt_out_smoothed.npy` for NaN absence.
4. Verify `sequence_summary.json` status distribution improves (fewer false `bad_loss`).

---

## 8) Rollout Sequence

## Phase A (stability hotfix)

- Implement Steps 1, 2, 3, 6, 7.
- Run synthetic reproductions from `research.md` to confirm bug closure.

## Phase B (behavior consistency)

- Implement Steps 8 and 9.
- Validate on multi-person/multi-run scenarios.

## Phase C (new lower-body feature)

- Implement Step 4 + Step 5.
- Validate hardcoded lower-body index tables against known static/lower-body-stable clips.

## Phase D (default tuning)

- Tune defaults:
  - `min_valid_points`
  - `zero_weight_strategy`
  - lower-body index-table coverage/accuracy for active pose dims (for example 133/204)
- Update `README.md` usage docs.

---

## 9) Detailed TODO Checklist

Use this as the execution checklist. Mark items complete during implementation.

## Phase A TODO (stability hotfix: Steps 1, 2, 3, 6, 7)

- [x] A1. Capture baseline behavior on one known problematic sample (current loss values, bad-frame tags, NaN presence in outputs).
- [x] A2. Add `min_valid_points` and `zero_weight_strategy` fields to `OptimizationConfig` in `optimize_mhr_pose.py`.
- [x] A3. Implement `_sanitize_subset_and_weights(...)` helper in `optimize_mhr_pose.py`.
- [x] A4. Apply sanitized valid-index filtering to per-view initialization scoring before Umeyama.
- [x] A5. Apply sanitized valid-index filtering in the main optimization loop before Umeyama and data loss.
- [x] A6. Add explicit handling for non-finite predicted subset points in both init scoring and optimization loop.
- [x] A7. Implement zero-weight fallback behavior (`uniform_finite`) and strict failure behavior (`fail`).
- [x] A8. Recompute post-restore `final_data_loss` and `final_loss` after `best_pose` is restored.
- [x] A9. Use recomputed post-restore losses for `classify_bad_optimization`.
- [x] A10. Update `smooth_frame_dict_sequence` in `video_temporal_utils.py` to treat only fully finite frames as valid smoothing anchors.
- [x] A11. Add optional finite guard before writing smoothed values back to frame dicts.
- [x] A12. Fix `.npy` payload handling in `triangulate_mhr3d_gt.py` to support direct ndarray payloads without `.item()`.
- [x] A13. Run targeted validations for B1/B2/B3/B4/B5 and record pass/fail notes.
- [x] A14. Re-run baseline problematic sample and compare to A1 metrics.

## Phase B TODO (behavior consistency: Steps 8, 9)

- [x] B1. Add Stage-1 person selection config fields in `sam3d_inference.py` (`person_select_strategy`, `person_index`).
- [x] B2. Update primary output extraction to support `first`, `largest_bbox`, and `person_index`.
- [x] B3. Confirm default Stage-1 strategy is `largest_bbox`.
- [x] B4. Add Stage-1 metadata write (`stage1_meta.json`) including input root and model identifiers.
- [x] B5. Implement Stage-1 metadata compatibility check in `run_full_pipeline.py` before reusing Stage-1 outputs.
- [x] B6. Ensure metadata mismatch forces Stage-1 rerun and does not silently reuse stale outputs.
- [x] B7. Validate multi-person behavior: default picks largest bbox; optional index picks requested person.
- [x] B8. Validate multi-run behavior: change in input/model config invalidates old Stage-1 cache.
- [x] B9. Record behavior deltas and confirm no regression on single-person runs.

## Phase C TODO (lower-body freeze feature: Steps 4, 5)

- [x] C1. Perform one-time mapping audit in `mhr70.py` and upstream pose mapping code to define lower-body keypoint set.
- [x] C2. Finalize lower-body keypoint-name list (hips, knees, ankles, toes, heels) and verify indices from `pose_info`.
- [x] C3. Derive hardcoded lower-body pose-parameter index table for each active pose dimension (at least 133 and 204 if used).
- [x] C4. Add `LOWER_BODY_POSE_IDXS_BY_DIM` table and document mapping provenance in code comments/docstring.
- [x] C5. Add `resolve_lower_body_pose_indices(pose_dim)` with explicit error on unsupported dimensions.
- [x] C6. Add `freeze_lower_body` field to `OptimizationConfig` and wire CLI arg `--freeze_lower_body`.
- [x] C7. Pass `--freeze_lower_body` through `run_full_pipeline.py` into Stage-3 optimization config.
- [x] C8. Build optimization mask by combining existing `keep_mask` with resolved lower-body frozen dimensions.
- [x] C9. Enforce freeze in backward/step path by zeroing gradients on frozen dims and clamping post-step pose values.
- [x] C10. Validate frozen dims remain numerically identical to initialization across all optimization iterations.
- [x] C11. Validate feature effect on static/lower-stable clips (reduced lower-body jitter without upper-body degradation).
- [x] C12. Validate failure mode for unsupported pose dimension is clear and actionable.

## Phase D TODO (default tuning + docs)

- [x] D1. Tune `min_valid_points` default using representative sequences with varying visibility/occlusion.
- [x] D2. Tune `zero_weight_strategy` default (`uniform_finite` vs `fail`) for best robustness/quality tradeoff.
- [x] D3. Review lower-body index table coverage/accuracy for all active pose dimensions used in production.
- [x] D5. Update `README.md` with final CLI/API behavior and examples for `--freeze_lower_body`.
- [x] D6. Add troubleshooting notes to `README.md` for new runtime errors (insufficient valid points, unsupported pose dim).
- [x] D7. Confirm Issue-D remains intentionally out of scope and document that status in release notes/changelog if used.
- [x] D8. Produce final before/after comparison summary: loss stability, bad-frame tagging quality, NaN incidence, lower-body jitter behavior.

## Global completion gates

- [x] G1. All Phase A/B/C/D checklist items completed.
- [x] G2. All `Definition of Done` criteria below are met.
- [x] G3. No unrelated files modified.

---

## 10) Definition of Done

Done when all are true:

1. All known reproductions from `research.md` are resolved.
2. No NaN propagation in smoothing output on finite-anchor sequences.
3. No false-perfect zero-loss on all-zero inlier masks.
4. Bad-loss classification uses post-restore final objective.
5. Lower-body freeze keeps configured lower-body pose dimensions unchanged through optimization.
6. New CLI/API behavior is documented, with only `--freeze_lower_body` exposed for this feature.
