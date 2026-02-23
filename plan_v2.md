# plan_v2

## Goal
Improve reconstruction quality for mostly static single-person videos (for example sign language) by combining:

1. SAM-3D-Body for full-body coverage,
2. WiLoR-mini (or HaMeR) for stronger hand keypoints,
3. MHR optimization with hand pose included.

At the same time, reduce runtime by removing expensive temporal terms and removing bad-frame retries.

## Scope and assumptions
- Single performer for the full sequence.
- Person is mostly stationary in global position.
- Motion is concentrated in hands and upper body.
- Camera calibration is available.

## WiLoR-mini facts verified from repo
Reference checked: `https://github.com/warmshao/WiLoR-mini` (local clone, commit `ebec42f`).

- Pipeline class: `WiLorHandPose3dEstimationPipeline`.
- Per detected hand, output is stored under `detect_rets[i]["wilor_preds"]`.
- Required keys for fusion:
  - `wilor_preds["pred_keypoints_2d"]` (21x2 in image pixels),
  - `wilor_preds["pred_keypoints_3d"]` (21x3),
  - hand side (`is_right`) from detector output.
- MANO/OpenPose hand order is explicitly used in WiLoR-mini:
  - `mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]`.
- Mapping from MANO/OpenPose 21 joints to MHR70 hand indices is documented in `hand_mapping.md`.

## Fixed v2 decisions

### Data fusion
- Keep SAM as base source for all 70 keypoints.
- Replace only hand keypoints with specialized hand model points when reliable.
- If specialized hand result is missing/weak, fall back to SAM hand keypoints.
- Keep torso and arm anchors (shoulders, elbows, wrists) stable; do not let noisy fingers control global alignment.

### Geometry
- Triangulate from mixed 2D keypoints:
  - hands from specialized model when available,
  - rest from SAM.

### Optimization
- Optimize both `body_pose_params` and `hand108`.
- Keep identity-like blocks fixed across sequence (`scale`, `shape`, `expr`; and other non-pose blocks from one reference frame/camera).
- Similarity alignment uses stable upper-body anchors only.
- Data supervision includes hands and upper body.

### Runtime
- Remove `L_vel` and `L_acc`.
- Keep `L_data`, `L_reg`, and optional light `L_temp`.
- Do not retry bad frames.
- If frame is bad, mark optimization output unusable and recover by interpolation.

## End-to-end v2 pipeline

1. Stage-1 base inference:
- Run SAM-3D-Body per camera and frame.
- Save baseline 70x2 keypoints per view.

2. Specialized hand inference:
- Run WiLoR-mini per camera frame.
- Extract `pred_keypoints_2d` and hand side (`is_right`) for each detected hand.

3. Hand detection association:
- Associate WiLoR left/right detections to subject hands.
- Prefer bbox proximity to SAM wrist/hand region and detection confidence.
- Keep only one detection per side per frame/camera.

4. 2D keypoint fusion:
- Convert WiLoR MANO/OpenPose 21 indices to MHR70 using `hand_mapping.md`.
- For each hand keypoint:
  - use specialized point if confidence and sanity checks pass,
  - else keep SAM point.
- Record source tag per keypoint: `specialized`, `sam_fallback`, or `missing`.

5. Triangulation:
- Run robust triangulation on fused 2D set.
- Keep inlier mask, reprojection error, and source metadata.
- Compute confidence per 3D point from inlier consistency.

6. Optimization:
- Variables: `body_pose_params` + `hand108`.
- Frozen: non-pose identity blocks and optionally lower body pose.
- Similarity anchors: shoulders, elbows, wrists (optional neck/acromion).
- Data term uses full supervised set with hand up-weighting when confidence is high.

7. Frame quality gate:
- Evaluate best/final total loss and best/final data loss.
- If frame is bad:
  - do not keep optimization output for that frame,
  - do not retry.

8. Recovery:
- Recover bad/missing frames by interpolation between nearest valid neighbors.
- If only one side exists, optionally one-sided copy with bounded span.

9. Export:
- Save optimized sequence, recovered sequence, and summary metrics.
- Include per-frame hand-source ratio and triangulation confidence statistics.

## Loss design (v2)

### Keep
- `L_data`: weighted robust 3D fitting loss.
- `L_reg`: regularization to initialization/reference.
- `L_temp` (optional, small): frame-to-frame smoothing.

### Remove
- `L_vel`, `L_acc`.

### Practical weighting policy
- Hand fingers: high weight only when triangulation confidence is strong.
- Wrists/elbows/shoulders: medium-high stable weights.
- Low-confidence points: down-weight or drop.

## Similarity alignment policy
- Alignment keypoints: only stable upper-body anchors.
- Supervision keypoints: broader set (hands + upper body subset).
- Scale: estimate once on a clean reference frame/window, then keep fixed for sequence to reduce drift.

## Bad-frame policy (no retry)
- Retry loop is disabled by design in v2.
- Any frame that fails quality gates is treated as invalid optimization output.
- Recovery is interpolation-first.
- Recovered frames stay explicitly labeled as recovered in summary.

## Implementation checklist by file

1. `run_full_pipeline.py`
- Add specialized hand inference toggle and path config.
- Add no-retry behavior switch (or hard disable retries in v2 mode).
- Ensure bad frame is not exported as normal optimized frame.

2. `sam3d_inference.py` (or new fusion helper module)
- Load WiLoR outputs.
- Associate left/right hands.
- Map MANO/OpenPose -> MHR indices.
- Fuse into final per-view `pred_keypoints_2d`.
- Save per-keypoint source tags.

3. `triangulate_mhr3d_gt.py`
- Consume fused 2D keypoints.
- Preserve source and inlier metadata.
- Output confidence weights for optimizer.

4. `optimize_mhr_pose.py`
- Add `hand108` to optimized variables.
- Freeze non-pose identity parameters from reference frame/camera.
- Remove `L_vel` and `L_acc` from v2 objective path.
- Use anchor-only set for similarity transform.

5. `README.md`
- Add v2 section with WiLoR integration and no-retry policy.

## Ablation and evaluation

### Accuracy
- SAM-only hands vs specialized+fallback hands.
- Optimize body-only vs body+hand108.
- Anchor-only similarity vs all-point similarity.

### Reliability
- Fraction of specialized hand points used per frame.
- Hand triangulation inlier ratio.
- Bad-frame rate and recovery footprint.

### Runtime
- Average optimization time/frame before vs after removing `L_vel` and `L_acc`.
- End-to-end sequence throughput.

## Expected outcome
- Better hand shape/pose fidelity than SAM-only.
- More stable identity and scale across long sequences.
- Lower runtime with simplified temporal objective and no retry loops.
