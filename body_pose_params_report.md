# body_pose_params_report

## Goal
Clarify what `mhr_model_params` means in SAM-3D-Body + MHR, and identify which parameter subset should be optimized when you want stable body identity and better hand tracking from specialized hand models (WiLoR, HaMeR, etc.).

---

## 1) What is `mhr_model_params`?

In SAM-3D-Body, `mhr_model_params` is the vector sent to MHR for mesh/joint generation.
It is built as:

- `model_params = concat(full_pose_params, scales)`

where:

- `full_pose_params`: 136 dims
- `scales`: 68 dims

So total:

- `mhr_model_params`: **204 dims**

### Exact construction inside `head.mhr_forward(...)`

The 204 vector is composed in this order:

1. `pose130 = body_pose_params[..., :130]`
2. `full_pose136 = concat(global_trans(3), global_rot(3), pose130(130))`
3. Hand insertion into pose block:
   - convert `hand108` to model-space hand joint params
   - write them into hand joint indices of `full_pose136`
4. `scales68 = scale_mean + scale28 @ scale_comps`
5. `mhr_model_params204 = concat(full_pose136, scales68)`

Therefore:

- `mhr_model_params` contains only:
  - global translation/rotation
  - pose block (after hand insertion)
  - scale/skeleton parameters
- `shape45` and `expr72` are **not** inside `mhr_model_params`; they are passed separately to MHR.

### Exact 204 layout

- `0:3` -> global translation (3)
- `3:6` -> global rotation (3)
- `6:136` -> local body pose used by MHR forward pass (130)
- `136:204` -> skeleton scale / limb-length-related parameters (68)

### Important: why this looks different from `body_pose_params (133D)`

There are **two related but different representations** in SAM-3D-Body:

- `body_pose_params` (saved as `body_pose` in outputs): **133D**
- `mhr_model_params` pose part: effectively uses **130 body dims** inside the 136 pose block

The flow is:

1. Network predicts continuous body pose (`pred_pose_cont`, 260D).
2. `compact_cont_to_model_params_body` converts it to `body_pose_params` (133D).
3. In `mhr_forward`, code applies `body_pose_params = body_pose_params[..., :130]`.
4. Then MHR pose block is formed as:
   - global trans (3) + global rot (3) + body pose used by forward (130) = 136.

So there is no contradiction:

- `133D` is an intermediate/output body-pose representation.
- `130D` is what is actually consumed in the final MHR forward path for `mhr_model_params`.

This matches the MHR paper split idea:

- `npose = 136`
- `nskel = 68`

with the practical meaning: skeleton/identity-like parameters should stay constant per performer/sequence.

---

## 2) Why body size drifts if all frame params are free?

If you optimize or re-estimate `mhr_model_params[136:204]` per frame, limb lengths and body proportions can change across time.
That creates visible inconsistency ("changing human length").

For a single person sequence, these 68 dims should be fixed (or estimated once from a clean reference frame and then frozen).

---

## 3) Hand-related pose dimensions inside the 136 pose block

SAM-3D-Body defines hand indices in 133-body space as:

- `mhr_param_hand_idxs = 62..115` (54 dims)

When converted to the 204 vector, add +6 offset (because 204 includes global trans+rot before body block):

- hand-related pose in 204 space: **68..121** (54 dims)

So:

- `mhr_model_params[68:122]` are hand-related pose entries.

---

## 4) Recommended optimization subsets

For single-person video + external hand model keypoints:

### Recommended default

- Freeze skeleton scales:
  - freeze `136:204`
- Freeze hand-related pose if hand is provided externally:
  - freeze `68:122`
- Optimize non-hand pose:
  - optimize `0:68` and `122:136`
  - optionally keep `0:3` translation fixed if world alignment is solved elsewhere

This gives stable body identity while allowing pose motion updates.

### Alternative presets

1. `full_pose_only`
- optimize `0:136`
- freeze `136:204`

2. `body_no_hand_pose_only` (best when hand detector is stronger)
- optimize `0:68` + `122:136`
- freeze `68:122` and `136:204`

3. `local_body_only`
- optimize `6:68` + `122:136`
- freeze `0:6`, `68:122`, `136:204`

---

## 5) Practical integration with current pipeline

Your current local optimizer mainly updates `body_pose_params` (133D) and already supports freezing non-pose blocks (`hand/scale/shape/expr`) when loading fixed params from a reference frame.

To move toward direct `mhr_model_params` subset optimization:

- define a 204-dim boolean mask
- set mask by preset (above)
- zero gradients for frozen dims each step
- clamp frozen dims back to reference values after optimizer step

This keeps sequence identity consistent while improving articulation fitting.

---

## 6) Source verification (official repos)

### SAM-3D-Body

- `mhr_head.py`: `body_pose_params = body_pose_params[..., :130]`
- `mhr_head.py`: `full_pose_params = cat([global_trans, global_rot, body_pose_params])` (136)
- `mhr_head.py`: `model_params = cat([full_pose_params, scales])` (204)
- `mhr_utils.py`: hand mask in body-133 space (`mhr_param_hand_idxs = 62..115`)

### MHR

- MHR README: model parameters are 204 and include pose/scaling behavior.
- Conversion utilities concatenate model parameters as `[trans, rot, pose, scale]` for LBS.

---

## 7) Final answer to your question

Yes: according to MHR’s formulation, you should treat the first 136 as pose-like, and for single-person sequence stability you typically optimize only pose (or a pose subset) while keeping the 68 skeleton-scale parameters fixed per sequence.
