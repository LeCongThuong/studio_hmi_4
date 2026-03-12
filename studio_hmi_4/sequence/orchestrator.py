"""End-to-end sequence orchestration with injectable stage hooks."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from studio_hmi_4.stage1.runner import Demo2Config
from studio_hmi_4.stage2.runner import TriangulationConfig
from studio_hmi_4.stage3.runner import OptimizationConfig

from .discovery import (
    cam_to_section_map,
    dir_has_min_cam_predictions,
    discover_frame_inputs,
    expected_stage1_meta,
    inject_numeric_gaps,
)
from .recovery import (
    load_fixed_body_pose_params,
    load_fixed_non_pose_mhr_params,
    load_similarity_if_good,
    load_stage1_meta,
    recover_missing_and_bad_frames,
    safe_scalar_float,
    safe_scalar_int,
    stage1_meta_matches,
    stale_frame_run_length,
    update_result_from_opt,
)
from .summary import save_resolved_config, save_summaries
from .temporal import (
    extract_keypoint_sequence,
    save_keypoint_sequence_mp4,
    save_npy_dict,
    smooth_frame_dict_sequence,
)
from .types import FramePipelineResult, FullPipelineConfig, FullPipelineResult


def run_full_pipeline(
    config: FullPipelineConfig,
    *,
    run_demo_fn: Callable,
    run_triangulation_fn: Callable,
    build_optimization_runtime_fn: Callable,
    run_optimization_fn: Callable,
) -> FullPipelineResult:
    image_root = Path(config.image_folder).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    inference_root = output_root / "inference"
    triangulation_root = output_root / "triangulation"
    optimization_root = output_root / "optimization"
    output_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(output_root=output_root, config=config)

    if not config.checkpoint_path:
        raise ValueError("--checkpoint_path is required.")

    min_views = max(2, int(config.min_views))
    cam_sections = cam_to_section_map(config.cams, config.toml_sections)
    opt_ckpt = config.opt_ckpt or (config.checkpoint_path if not config.hf_repo else None)
    opt_mhr_pt = config.opt_mhr_pt or config.mhr_path
    if not (config.hf_repo or opt_ckpt):
        raise ValueError("Optimization requires --hf_repo or --opt_ckpt.")

    inferred_npy_root = (inference_root / "npy").resolve()
    stage1_meta_path = (inference_root / "stage1_meta.json").resolve()
    expected_meta = expected_stage1_meta(config=config, image_root=image_root)
    reuse_stage1 = False
    if config.skip_inference:
        if not inferred_npy_root.is_dir():
            raise FileNotFoundError(
                "Requested --skip_inference but existing stage-1 output root was not found: "
                f"{inferred_npy_root}"
            )
        npy_root = inferred_npy_root
        print(f"[PIPELINE] Skipping stage-1 inference and reusing existing outputs at: {npy_root}")
    else:
        if not config.overwrite and config.frame_rel is not None and inferred_npy_root.is_dir():
            rel_dir_path = (inferred_npy_root / config.frame_rel).resolve()
            meta_matches = stage1_meta_matches(
                existing_meta=load_stage1_meta(stage1_meta_path),
                expected_meta=expected_meta,
            )
            if rel_dir_path.is_dir() and dir_has_min_cam_predictions(rel_dir_path, config.cams, min_views=min_views) and meta_matches:
                reuse_stage1 = True

        if reuse_stage1:
            npy_root = inferred_npy_root
            print(f"[PIPELINE] Reusing existing stage-1 outputs at: {npy_root}")
        else:
            demo_cfg = Demo2Config(
                image_folder=str(image_root),
                output_folder=str(inference_root),
                checkpoint_path=config.checkpoint_path,
                detector_name=config.detector_name,
                segmentor_name=config.segmentor_name,
                fov_name=config.fov_name,
                detector_path=config.detector_path,
                segmentor_path=config.segmentor_path,
                fov_path=config.fov_path,
                mhr_path=config.mhr_path,
                bbox_thresh=config.bbox_thresh,
                use_mask=config.use_mask,
                debug=config.debug_inference,
                save_mhr_params=config.save_mhr_params,
                include_rel_dirs=[config.frame_rel] if config.frame_rel else None,
                person_select_strategy=config.person_select_strategy,
                person_index=config.person_index,
                enable_specialized_hand_fusion=config.enable_specialized_hand_fusion,
                specialized_hand_source=config.specialized_hand_source,
                specialized_hand_model=config.specialized_hand_model,
                specialized_hand_input_root=config.specialized_hand_input_root,
                specialized_hand_device=config.specialized_hand_device,
                specialized_hand_detector_conf=config.specialized_hand_detector_conf,
                specialized_hand_rescale_factor=config.specialized_hand_rescale_factor,
                specialized_hand_wrist_max_dist_px=config.specialized_hand_wrist_max_dist_px,
                replace_wrist_with_specialized=config.replace_wrist_with_specialized,
                specialized_hand_debug_vis=config.specialized_hand_debug_vis,
                specialized_hand_debug_dirname=config.specialized_hand_debug_dirname,
                specialized_hand_verbose=config.specialized_hand_verbose,
                wilor_pretrained_dir=config.wilor_pretrained_dir,
                wilor_repo_id=config.wilor_repo_id,
            )
            demo_result = run_demo_fn(demo_cfg)
            npy_root = demo_result.npy_root.resolve()

    frame_inputs = discover_frame_inputs(npy_root=npy_root, cams=config.cams, frame_rel=config.frame_rel)
    if config.frame_rel is None:
        frame_inputs = inject_numeric_gaps(frame_inputs)
    if not frame_inputs:
        raise FileNotFoundError(f"No frame directories with camera predictions {config.cams} found under {npy_root}")

    fixed_hand_params = None
    fixed_scale_params = None
    fixed_shape_params = None
    fixed_expr_params = None
    fixed_lower_body_pose_params = None

    def _find_reference_entry(frame_idx: int, flag_name: str):
        ref_entry = next(
            (
                entry
                for entry in frame_inputs
                if entry.frame_index is not None and int(entry.frame_index) == int(frame_idx) and entry.npy_dir is not None
            ),
            None,
        )
        if ref_entry is None or ref_entry.npy_dir is None:
            raise FileNotFoundError(
                f"Could not find frame index {int(frame_idx)} with available npy inputs for {flag_name}."
            )
        return ref_entry

    if config.fixed_mhr_param_frame_idx is not None:
        ref_idx = int(config.fixed_mhr_param_frame_idx)
        ref_entry = _find_reference_entry(ref_idx, "--fixed_mhr_param_frame_idx")
        fixed_cam = str(config.fixed_mhr_param_cam).strip()
        if fixed_cam == "":
            raise ValueError("--fixed_mhr_param_cam cannot be empty.")
        (
            fixed_hand_params,
            fixed_scale_params,
            fixed_shape_params,
            fixed_expr_params,
        ) = load_fixed_non_pose_mhr_params(
            npy_dir=ref_entry.npy_dir,
            cam=fixed_cam,
            include_hand_pose=False,
        )
        print(
            "[PIPELINE] Using fixed non-pose MHR params from "
            f"frame_index={ref_idx} rel='{ref_entry.rel_dir or '.'}' cam='{fixed_cam}'."
        )
        print("[PIPELINE] Hand pose remains framewise optimized (not fixed from reference).")

    if config.fixed_lower_body_pose_frame_idx is not None:
        ref_idx = int(config.fixed_lower_body_pose_frame_idx)
        ref_entry = _find_reference_entry(ref_idx, "--fixed_lower_body_pose_frame_idx")
        fixed_cam = str(config.fixed_lower_body_pose_cam).strip()
        if fixed_cam == "":
            raise ValueError("--fixed_lower_body_pose_cam cannot be empty.")
        fixed_lower_body_pose_params = load_fixed_body_pose_params(
            npy_dir=ref_entry.npy_dir,
            cam=fixed_cam,
        )
        print(
            "[PIPELINE] Using fixed lower-body pose template from "
            f"frame_index={ref_idx} rel='{ref_entry.rel_dir or '.'}' cam='{fixed_cam}'."
        )
        if config.freeze_lower_body:
            print(
                "[PIPELINE] --fixed_lower_body_pose_frame_idx is active; "
                "reference lower-body template overrides --freeze_lower_body temporal lower-body freezing."
            )

    runtime_seed = next((entry for entry in frame_inputs if entry.npy_dir is not None and len(entry.available_cams) >= min_views), None)
    if runtime_seed is None or runtime_seed.npy_dir is None:
        raise FileNotFoundError(
            f"No frame has enough views (>= {min_views}) for optimization runtime initialization."
        )
    runtime_cfg = OptimizationConfig(
        npz=Path("runtime.npz"),
        npy_dir=runtime_seed.npy_dir,
        cams=list(runtime_seed.available_cams),
        out_npy=Path("runtime.npy"),
        debug_dir=(optimization_root / "debug_opt"),
        hf_repo=config.hf_repo,
        ckpt=opt_ckpt,
        mhr_pt=opt_mhr_pt,
        device=config.device,
        save_debug_artifacts=False,
        min_valid_points=config.min_valid_points,
        zero_weight_strategy=config.zero_weight_strategy,
        freeze_lower_body=bool(config.freeze_lower_body and fixed_lower_body_pose_params is None),
    )
    opt_runtime = build_optimization_runtime_fn(runtime_cfg)

    prev_good_pose = None
    prev_good_sim_scale = None
    prev_good_sim_R = None
    prev_good_sim_t = None
    frame_results: list[FramePipelineResult] = []

    for idx, entry in enumerate(frame_inputs, start=1):
        rel_dir = entry.rel_dir
        print(f"[PIPELINE] Frame {idx}/{len(frame_inputs)} rel='{rel_dir or '.'}'")

        fr = FramePipelineResult(
            rel_dir=rel_dir,
            frame_index=entry.frame_index,
            npy_dir=entry.npy_dir,
            available_cams=list(entry.available_cams),
            used_cams=[],
            triangulated_npz=None,
            optimized_npy=None,
            smoothed_npy=None,
            status="pending",
        )

        if entry.npy_dir is None:
            fr.status = "missing_input"
            frame_results.append(fr)
            continue

        used_cams = list(entry.available_cams)
        fr.used_cams = used_cams
        if len(used_cams) < min_views:
            fr.status = "insufficient_views"
            frame_results.append(fr)
            continue

        tri_out = (triangulation_root / rel_dir / config.triangulated_name).resolve()
        fr.triangulated_npz = tri_out
        tri_out.parent.mkdir(parents=True, exist_ok=True)
        tri_debug_dir = (tri_out.parent / "debug") if config.save_triangulation_debug else None
        rel_img_dir = image_root / rel_dir
        img_dir = rel_img_dir if rel_img_dir.is_dir() else image_root

        if config.skip_triangulation:
            if not tri_out.exists():
                fr.status = "triangulation_missing"
                fr.error = f"Requested --skip_triangulation but file was not found: {tri_out}"
                print(f"[PIPELINE][WARN] {fr.error}")
                frame_results.append(fr)
                continue
            print(f"[PIPELINE] Skipping triangulation and reusing existing output: {tri_out}")
        elif tri_out.exists() and not config.overwrite:
            print(f"[PIPELINE] Reusing triangulation: {tri_out}")
        else:
            try:
                tri_cfg = TriangulationConfig(
                    mhr_py=config.mhr_py,
                    caliscope_toml=config.caliscope_toml,
                    cams=used_cams,
                    toml_sections=[cam_sections[cam] for cam in used_cams],
                    npy_dir=str(entry.npy_dir),
                    out_npz=str(tri_out),
                    normalized=config.normalized,
                    pixel=config.pixel,
                    invert_extrinsics=config.invert_extrinsics,
                    lm_iters=config.lm_iters,
                    lm_lambda=config.lm_lambda,
                    lm_eps=config.lm_eps,
                    debug=False,
                    debug_dir=str(tri_debug_dir) if tri_debug_dir else None,
                    img_dir=str(img_dir),
                    score_type=config.score_type,
                    huber_delta=config.huber_delta,
                    inlier_thresh=config.inlier_thresh,
                    robust_lm=config.robust_lm,
                    robust_lm_delta=config.robust_lm_delta,
                )
                run_triangulation_fn(tri_cfg)
            except Exception as exc:
                fr.status = "triangulation_failed"
                fr.error = f"{type(exc).__name__}: {exc}"
                print(f"[PIPELINE][WARN] Triangulation failed at '{rel_dir}': {fr.error}")
                traceback.print_exc()
                frame_results.append(fr)
                continue

        optimized_npy = (optimization_root / rel_dir / config.optimized_name).resolve()
        fr.optimized_npy = optimized_npy

        if optimized_npy.exists() and not config.overwrite:
            print(f"[PIPELINE] Reusing optimized output: {optimized_npy}")
            try:
                from .temporal import load_npy_dict

                d = load_npy_dict(optimized_npy)
                is_bad = bool(int(np.asarray(d.get("opt_is_bad_loss", 0)).reshape(())))
                is_recovered = bool(int(np.asarray(d.get("opt_recovered", 0)).reshape(())))

                pose = None
                if (not is_bad) and ("body_pose_params" in d):
                    try:
                        pose = np.asarray(d["body_pose_params"], dtype=np.float32).reshape(-1)
                    except Exception:
                        pose = None

                if pose is not None:
                    if is_recovered:
                        mode = str(d.get("opt_recovery_mode", "")).strip()
                        if mode == "copy_prev":
                            fr.status = "recovered_copy_prev"
                        elif mode == "copy_next":
                            fr.status = "recovered_copy_next"
                        elif mode == "interpolate":
                            fr.status = "recovered_interpolated"
                        else:
                            fr.status = "recovered"
                        rec_from = str(d.get("opt_recovered_from", "")).strip()
                        fr.recovered_from = rec_from or None
                    else:
                        fr.status = "ok"
                        prev_good_pose = np.asarray(pose, dtype=np.float32).reshape(-1).copy()
                        sim = load_similarity_if_good(optimized_npy)
                        if sim is not None:
                            prev_good_sim_scale = float(sim[0])
                            prev_good_sim_R = sim[1].copy()
                            prev_good_sim_t = sim[2].copy()
                else:
                    fr.status = "bad_loss"
                    fr.is_bad_loss = True

                if is_recovered:
                    fr.best_loss = None
                    fr.final_loss = None
                    fr.best_data_loss = None
                    fr.final_data_loss = None
                    fr.best_iter = None
                    fr.is_bad_loss = False
                else:
                    fr.best_loss = safe_scalar_float(d, "opt_best_loss")
                    fr.final_loss = safe_scalar_float(d, "opt_final_loss")
                    fr.best_data_loss = safe_scalar_float(d, "opt_best_data_loss")
                    fr.final_data_loss = safe_scalar_float(d, "opt_final_data_loss")
                    fr.best_iter = safe_scalar_int(d, "opt_best_iter")
                    if "opt_is_bad_loss" in d:
                        fr.is_bad_loss = bool(int(np.asarray(d["opt_is_bad_loss"]).reshape(())))
                        if fr.is_bad_loss:
                            fr.status = "bad_loss"
            except Exception:
                pass
            frame_results.append(fr)
            continue

        stale_run = stale_frame_run_length(frame_results)
        temporal_guard_enabled = int(config.max_stale_temporal_frames) > 0
        use_temporal_priors = not (temporal_guard_enabled and stale_run >= int(config.max_stale_temporal_frames))
        if not use_temporal_priors:
            print(f"[PIPELINE] Temporal priors disabled at '{rel_dir}' after {stale_run} consecutive non-good frames.")

        init_prev_body_pose = None if (not use_temporal_priors or prev_good_pose is None) else prev_good_pose.copy()
        init_prev_sim_scale = None if (not use_temporal_priors) else prev_good_sim_scale
        init_prev_sim_R = None if (not use_temporal_priors or prev_good_sim_R is None) else prev_good_sim_R.copy()
        init_prev_sim_t = None if (not use_temporal_priors or prev_good_sim_t is None) else prev_good_sim_t.copy()

        try:
            opt_cfg = OptimizationConfig(
                npz=tri_out,
                npy_dir=entry.npy_dir,
                cams=used_cams,
                out_npy=optimized_npy,
                debug_dir=(optimization_root / rel_dir / "debug_opt").resolve(),
                hf_repo=config.hf_repo,
                ckpt=opt_ckpt,
                mhr_pt=opt_mhr_pt,
                device=config.device,
                iters=config.iters,
                lr=config.lr,
                with_scale=config.with_scale,
                huber_m=config.huber_m,
                w_pose_reg=config.w_pose_reg,
                w_hand_reg=config.w_hand_reg,
                w_temporal=config.w_temporal,
                temporal_init_blend=config.temporal_init_blend,
                init_prev_body_pose=init_prev_body_pose,
                init_prev_sim_scale=init_prev_sim_scale,
                init_prev_sim_R=init_prev_sim_R,
                init_prev_sim_t=init_prev_sim_t,
                fixed_hand_pose_params=None if fixed_hand_params is None else fixed_hand_params.copy(),
                fixed_scale_params=None if fixed_scale_params is None else fixed_scale_params.copy(),
                fixed_shape_params=None if fixed_shape_params is None else fixed_shape_params.copy(),
                fixed_expr_params=None if fixed_expr_params is None else fixed_expr_params.copy(),
                fixed_lower_body_pose_params=(
                    None if fixed_lower_body_pose_params is None else fixed_lower_body_pose_params.copy()
                ),
                optimize_hand_pose=bool(config.optimize_hand_pose),
                use_anchor_similarity=bool(config.use_anchor_similarity),
                bad_loss_threshold=config.bad_loss_threshold,
                bad_data_loss_threshold=config.bad_data_loss_threshold,
                bad_loss_growth_ratio=config.bad_loss_growth_ratio,
                min_valid_points=config.min_valid_points,
                zero_weight_strategy=config.zero_weight_strategy,
                freeze_lower_body=bool(config.freeze_lower_body and fixed_lower_body_pose_params is None),
                topk_print=config.topk_print,
                save_debug_artifacts=config.save_opt_debug,
            )
            opt_res = run_optimization_fn(opt_cfg, runtime=opt_runtime)
            update_result_from_opt(fr, opt_res)
            fr.status = "bad_loss" if opt_res.is_bad_loss else "ok"
            fr.is_bad_loss = bool(opt_res.is_bad_loss)
            if not opt_res.is_bad_loss:
                prev_good_pose = np.asarray(opt_res.best_pose, dtype=np.float32).reshape(-1).copy()
                prev_good_sim_scale = float(opt_res.sim_scale)
                prev_good_sim_R = np.asarray(opt_res.sim_R, dtype=np.float32).copy()
                prev_good_sim_t = np.asarray(opt_res.sim_t, dtype=np.float32).copy()
        except Exception as exc:
            fr.status = "optimization_failed"
            fr.error = f"{type(exc).__name__}: {exc}"
            print(f"[PIPELINE][WARN] Optimization failed at '{rel_dir}': {fr.error}")
            traceback.print_exc()
            frame_results.append(fr)
            continue

        frame_results.append(fr)

    frame_dicts = recover_missing_and_bad_frames(
        frame_results=frame_results,
        optimization_root=optimization_root,
        optimized_name=config.optimized_name,
        max_edge_copy_span=config.max_edge_recovery_copy_span,
    )

    smoothed_dicts = smooth_frame_dict_sequence(
        frame_dicts,
        alpha=config.smoothing_alpha,
        median_window=config.smoothing_median_window,
        outlier_sigma=config.smoothing_outlier_sigma,
    )
    for fr, smoothed_entry in zip(frame_results, smoothed_dicts):
        if smoothed_entry is None:
            continue
        smoothed_out = (optimization_root / fr.rel_dir / config.smoothed_name).resolve()
        save_npy_dict(smoothed_out, smoothed_entry)
        fr.smoothed_npy = smoothed_out

    if config.save_sequence_mp4:
        points_seq = extract_keypoint_sequence(smoothed_dicts)
        if points_seq.shape[0] > 0:
            mp4_path = (output_root / config.sequence_mp4_name).resolve()
            ok = save_keypoint_sequence_mp4(points_seq, mp4_path, fps=config.sequence_fps, title="4D Reconstruction")
            if ok:
                print(f"[PIPELINE] Saved sequence debug mp4: {mp4_path}")
            else:
                print("[PIPELINE][WARN] Failed to save sequence debug mp4.")

    summary_json = None
    if config.save_summary_json:
        summary_json = save_summaries(output_root=output_root, npy_root=npy_root, frame_results=frame_results)
        print(f"[PIPELINE] Saved summary: {summary_json}")

    return FullPipelineResult(
        output_root=output_root,
        npy_root=npy_root,
        frames=frame_results,
        summary_json=summary_json,
    )
