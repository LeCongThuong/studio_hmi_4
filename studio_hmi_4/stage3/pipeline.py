"""Core optimization pipeline for stage-3."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from studio_hmi_4.common import (
    validate_optimization_result_dict,
    validate_stage1_prediction_dict,
    validate_triangulation_bundle,
    write_ply,
)

from .alignment import (
    build_alignment_anchor_local_indices,
    build_base_keep_mask,
    classify_bad_optimization,
    huber,
    resolve_lower_body_pose_indices,
    resolve_valid_indices_for_prediction,
    safe_growth,
    sanitize_subset_and_weights,
    select_alignment_subset_tensors,
    umeyama_similarity,
)
from .debug import plot_3d_compare, plot_loss_curve
from .runtime import (
    apply_repo_camera_flip_xyz,
    build_optimization_runtime,
    ensure_dir,
    find_npy_for_cam,
    get_param_array,
    mhr_fk,
    safe_numpy_item_load,
    to_torch,
)
from .types import OptimizationConfig, OptimizationRunResult, OptimizationRuntime


def run_optimization(
    config: OptimizationConfig,
    runtime: Optional[OptimizationRuntime] = None,
) -> OptimizationRunResult:
    """Run optimization of body pose parameters using refined multi-view 3D points."""

    if not config.hf_repo and not config.ckpt:
        raise ValueError("Either hf_repo or ckpt must be provided.")

    if runtime is None:
        runtime = build_optimization_runtime(config)

    device = runtime.device
    head = runtime.head
    hand_mask = runtime.hand_mask
    runtime_keep_mask = runtime.keep_mask

    debug_dir = config.debug_dir.expanduser().resolve()
    if config.save_debug_artifacts:
        ensure_dir(debug_dir)
    out_npy = config.out_npy.expanduser().resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    cams = list(config.cams)
    npy_dir = config.npy_dir.expanduser().resolve()
    npz_path = config.npz.expanduser().resolve()

    z = np.load(npz_path, allow_pickle=True)
    tri_contract = validate_triangulation_bundle(z)
    subset_idx = tri_contract.subset_indices.astype(np.int64).reshape(-1)
    subset_names = tri_contract.subset_names
    gtM = tri_contract.points3d_refined.astype(np.float32)
    M = gtM.shape[0]

    if tri_contract.inlier_mask is not None:
        wM = tri_contract.inlier_mask.astype(np.float32)
        if wM.ndim == 2:
            wM = wM.mean(axis=1)
        wM = np.clip(wM, 0.0, 1.0)
    else:
        wM = np.ones((M,), dtype=np.float32)

    finite_gt_mask_np, wM_np, _ = sanitize_subset_and_weights(
        gtM=gtM,
        wM=wM,
        min_valid_points=config.min_valid_points,
        strategy=config.zero_weight_strategy,
    )
    finite_gt_mask_t = torch.from_numpy(finite_gt_mask_np).to(device=device, dtype=torch.bool)
    gtM_t = to_torch(gtM, device)
    wM_t = to_torch(wM_np, device)
    anchor_local_idx_np = (
        build_alignment_anchor_local_indices(subset_names)
        if bool(config.use_anchor_similarity)
        else np.zeros((0,), dtype=np.int64)
    )
    anchor_local_idx_t: Optional[torch.Tensor] = None
    if int(anchor_local_idx_np.size) > 0:
        anchor_local_idx_t = torch.from_numpy(anchor_local_idx_np).to(device=device, dtype=torch.long)

    view_scores_3d: Dict[str, float] = {}
    view_mean_px: Dict[str, float] = {}
    view_align: Dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    view_dicts: Dict[str, dict] = {}

    for cam in cams:
        key = f"{cam}_mean_err_px_refined"
        view_mean_px[cam] = float(np.asarray(z[key]).reshape(())) if key in z else float("inf")

    for cam in cams:
        npy_path = find_npy_for_cam(npy_dir, cam)
        d = safe_numpy_item_load(npy_path)
        validate_stage1_prediction_dict(d, require_pose_blocks=True)
        view_dicts[cam] = d

        pose133 = to_torch(d["body_pose_params"], device).flatten().to(torch.float32)
        hand108 = get_param_array(d, "hand_pose_params", device)
        scale28 = get_param_array(d, "scale_params", device)
        shape45 = get_param_array(d, "shape_params", device)
        expr72 = get_param_array(d, "expr_params", device)

        view_keep_mask = build_base_keep_mask(
            pose_dim=int(pose133.numel()),
            hand_mask_133=hand_mask,
            device=device,
        )
        pose_eff = pose133 * view_keep_mask

        out = mhr_fk(
            head,
            pose_eff,
            hand108,
            scale28,
            shape45,
            expr72,
            device,
            want_verts=False,
            want_joint=False,
            want_model_params=False,
        )
        keypoints_308 = out[1].squeeze(0)
        k70 = apply_repo_camera_flip_xyz(keypoints_308[:70])
        predM = k70[subset_idx]
        try:
            valid_idx_t, wM_view_t = resolve_valid_indices_for_prediction(
                predM=predM,
                finite_gt_mask_t=finite_gt_mask_t,
                base_wM_t=wM_t,
                min_valid_points=config.min_valid_points,
                strategy=config.zero_weight_strategy,
            )
        except RuntimeError:
            view_scores_3d[cam] = float("inf")
            view_align[cam] = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            print(f"[score] {cam:>8s}: 3D=inf m   mean_px={view_mean_px[cam]:.3f}   file={npy_path.name}")
            continue

        predM_v = predM.index_select(0, valid_idx_t)
        gtM_v = gtM_t.index_select(0, valid_idx_t)
        wM_v = wM_view_t.index_select(0, valid_idx_t)
        pred_align_v, gt_align_v, w_align_v = select_alignment_subset_tensors(
            predM_v=predM_v,
            gtM_v=gtM_v,
            wM_v=wM_v,
            valid_idx_t=valid_idx_t,
            anchor_local_idx_t=anchor_local_idx_t,
        )
        s, R, t = umeyama_similarity(pred_align_v, gt_align_v, w=w_align_v, with_scale=config.with_scale)
        predM_aligned_v = s * (predM_v @ R.T) + t[None, :]
        r = torch.sqrt(((predM_aligned_v - gtM_v) ** 2).sum(dim=1) + 1e-12)
        score = (wM_v * r).sum() / (wM_v.sum() + 1e-9)

        view_scores_3d[cam] = float(score.detach().cpu().item())
        view_align[cam] = (float(s.cpu().item()), R.cpu().numpy(), t.cpu().numpy())

        print(f"[score] {cam:>8s}: 3D={view_scores_3d[cam]:.6f} m   mean_px={view_mean_px[cam]:.3f}   file={npy_path.name}")

    best_cam = sorted(cams, key=lambda c: (view_scores_3d[c], view_mean_px[c]))[0]
    print(f"\n[init] best_cam = {best_cam}  (3D={view_scores_3d[best_cam]:.6f} m, mean_px={view_mean_px[best_cam]:.3f})")

    init_dict = view_dicts[best_cam]
    init_pose_raw = to_torch(init_dict["body_pose_params"], device).flatten().to(torch.float32)
    init_hand = get_param_array(init_dict, "hand_pose_params", device)
    init_scale = get_param_array(init_dict, "scale_params", device)
    init_shape = get_param_array(init_dict, "shape_params", device)
    init_expr = get_param_array(init_dict, "expr_params", device)

    if config.fixed_hand_pose_params is not None:
        fixed_hand = to_torch(config.fixed_hand_pose_params, device).flatten().to(torch.float32)
        if int(fixed_hand.numel()) != int(init_hand.numel()):
            raise ValueError(
                "fixed_hand_pose_params dim mismatch: "
                f"expected {int(init_hand.numel())}, got {int(fixed_hand.numel())}"
            )
        init_hand = fixed_hand
    if config.fixed_scale_params is not None:
        fixed_scale = to_torch(config.fixed_scale_params, device).flatten().to(torch.float32)
        if int(fixed_scale.numel()) != int(init_scale.numel()):
            raise ValueError(
                "fixed_scale_params dim mismatch: "
                f"expected {int(init_scale.numel())}, got {int(fixed_scale.numel())}"
            )
        init_scale = fixed_scale
    if config.fixed_shape_params is not None:
        fixed_shape = to_torch(config.fixed_shape_params, device).flatten().to(torch.float32)
        if int(fixed_shape.numel()) != int(init_shape.numel()):
            raise ValueError(
                "fixed_shape_params dim mismatch: "
                f"expected {int(init_shape.numel())}, got {int(fixed_shape.numel())}"
            )
        init_shape = fixed_shape
    if config.fixed_expr_params is not None:
        fixed_expr = to_torch(config.fixed_expr_params, device).flatten().to(torch.float32)
        if int(fixed_expr.numel()) != int(init_expr.numel()):
            raise ValueError(
                "fixed_expr_params dim mismatch: "
                f"expected {int(init_expr.numel())}, got {int(fixed_expr.numel())}"
            )
        init_expr = fixed_expr

    pose_dim = int(init_pose_raw.numel())
    if int(runtime_keep_mask.numel()) == pose_dim:
        base_keep_mask = runtime_keep_mask.to(device=device, dtype=torch.float32)
    else:
        base_keep_mask = build_base_keep_mask(
            pose_dim=pose_dim,
            hand_mask_133=hand_mask,
            device=device,
        )

    use_fixed_lower_body_pose = config.fixed_lower_body_pose_params is not None
    lock_lower_body_pose = bool(config.freeze_lower_body) or use_fixed_lower_body_pose
    lower_idxs_t: Optional[torch.Tensor] = None
    optimize_mask = base_keep_mask.clone()
    if lock_lower_body_pose:
        lower_idxs_np = resolve_lower_body_pose_indices(pose_dim=pose_dim)
        if lower_idxs_np.size > 0:
            lower_idxs_t = torch.from_numpy(lower_idxs_np).to(device=device, dtype=torch.long)
            optimize_mask[lower_idxs_t] = 0.0

    init_pose_ref = init_pose_raw.clone()
    init_hand_ref = init_hand.clone()
    frozen_pose_target = init_pose_raw.clone()
    if use_fixed_lower_body_pose:
        fixed_lower_body_pose = to_torch(config.fixed_lower_body_pose_params, device).flatten().to(torch.float32)
        if int(fixed_lower_body_pose.numel()) != pose_dim:
            raise ValueError(
                "fixed_lower_body_pose_params dim mismatch: "
                f"expected {pose_dim}, got {int(fixed_lower_body_pose.numel())}"
            )
        if lower_idxs_t is not None and int(lower_idxs_t.numel()) > 0:
            frozen_pose_target.index_copy_(
                0,
                lower_idxs_t,
                fixed_lower_body_pose.index_select(0, lower_idxs_t),
            )
    temporal_prev_np = config.init_prev_body_pose
    if temporal_prev_np is None:
        temporal_prev_np = config.init_body_pose
    temporal_prev_prev_np = config.init_prev_prev_body_pose

    temporal_pose = None
    temporal_prev_prev_pose = None
    temporal_velocity_target = None
    used_temporal_init = temporal_prev_np is not None
    if temporal_prev_np is not None:
        temporal_pose = to_torch(temporal_prev_np, device).flatten().to(torch.float32)
        if int(temporal_pose.numel()) != pose_dim:
            raise ValueError(f"Temporal pose dim mismatch: expected {pose_dim}, got {int(temporal_pose.numel())}")
        init_target = temporal_pose
        if temporal_prev_prev_np is not None:
            temporal_prev_prev_pose = to_torch(temporal_prev_prev_np, device).flatten().to(torch.float32)
            if int(temporal_prev_prev_pose.numel()) != pose_dim:
                raise ValueError(
                    f"Temporal prev-prev pose dim mismatch: expected {pose_dim}, got {int(temporal_prev_prev_pose.numel())}"
                )
            extrap = float(np.clip(config.temporal_extrapolation, 0.0, 2.0))
            temporal_velocity_target = temporal_pose + extrap * (temporal_pose - temporal_prev_prev_pose)
            init_target = temporal_velocity_target
        blend = float(np.clip(config.temporal_init_blend, 0.0, 1.0))
        blend_mask = optimize_mask * blend
        init_pose_ref = init_pose_ref * (1.0 - blend_mask) + init_target * blend_mask
        if config.freeze_lower_body and not use_fixed_lower_body_pose:
            frozen_pose_target = torch.where(
                optimize_mask == 0,
                temporal_pose,
                frozen_pose_target,
            )

    optimize_hand_pose = bool(config.optimize_hand_pose) and (config.fixed_hand_pose_params is None)

    pose = init_pose_ref.clone().detach().requires_grad_(True)
    hand = init_hand_ref.clone().detach()
    if optimize_hand_pose:
        hand.requires_grad_(True)
    opt_params: List[torch.Tensor] = [pose]
    if optimize_hand_pose:
        opt_params.append(hand)
    opt = torch.optim.Adam(opt_params, lr=config.lr)

    loss_hist: List[float] = []
    data_loss_hist: List[float] = []

    if config.save_debug_artifacts:
        with torch.no_grad():
            pose_eff0 = pose.detach() * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)
            out0 = mhr_fk(
                head,
                pose_eff0,
                hand.detach(),
                init_scale,
                init_shape,
                init_expr,
                device,
                want_verts=False,
                want_joint=False,
                want_model_params=False,
            )
            k70_0 = apply_repo_camera_flip_xyz(out0[1].squeeze(0)[:70])
            predM0 = k70_0[subset_idx]
            valid_idx0_t, wM0_t = resolve_valid_indices_for_prediction(
                predM=predM0,
                finite_gt_mask_t=finite_gt_mask_t,
                base_wM_t=wM_t,
                min_valid_points=config.min_valid_points,
                strategy=config.zero_weight_strategy,
            )
            predM0_v = predM0.index_select(0, valid_idx0_t)
            gtM0_v = gtM_t.index_select(0, valid_idx0_t)
            wM0_v = wM0_t.index_select(0, valid_idx0_t)
            pred0_align_v, gt0_align_v, w0_align_v = select_alignment_subset_tensors(
                predM_v=predM0_v,
                gtM_v=gtM0_v,
                wM_v=wM0_v,
                valid_idx_t=valid_idx0_t,
                anchor_local_idx_t=anchor_local_idx_t,
            )
            s0, R0, t0 = umeyama_similarity(pred0_align_v, gt0_align_v, w=w0_align_v, with_scale=config.with_scale)
            predM0_al = s0 * (predM0_v @ R0.T) + t0[None, :]
            plot_3d_compare(
                gtM[valid_idx0_t.cpu().numpy()],
                predM0_al.cpu().numpy(),
                f"Init ({best_cam}) aligned to GT (M={M})",
                debug_dir / "compare_init_subset.png",
            )

    min_iters = max(1, min(int(config.min_iters), int(config.iters)))
    patience = max(1, int(config.early_stop_patience))
    improve_tol = max(0.0, float(config.early_stop_tol))
    divergence_ratio = max(1.0, float(config.loss_divergence_ratio))
    best_loss = float("inf")
    best_data_loss = float("inf")
    best_iter = -1
    best_pose = pose.detach().clone()
    best_hand = hand.detach().clone()
    no_improve = 0

    for it in range(config.iters):
        opt.zero_grad(set_to_none=True)
        pose_eff = pose * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)

        out = mhr_fk(
            head,
            pose_eff,
            hand,
            init_scale,
            init_shape,
            init_expr,
            device,
            want_verts=False,
            want_joint=False,
            want_model_params=False,
        )
        k70 = apply_repo_camera_flip_xyz(out[1].squeeze(0)[:70])
        predM = k70[subset_idx]

        valid_idx_t, wM_masked_t = resolve_valid_indices_for_prediction(
            predM=predM,
            finite_gt_mask_t=finite_gt_mask_t,
            base_wM_t=wM_t,
            min_valid_points=config.min_valid_points,
            strategy=config.zero_weight_strategy,
        )
        predM_v = predM.index_select(0, valid_idx_t)
        gtM_v = gtM_t.index_select(0, valid_idx_t)
        wM_v = wM_masked_t.index_select(0, valid_idx_t)

        with torch.no_grad():
            pred_align_v, gt_align_v, w_align_v = select_alignment_subset_tensors(
                predM_v=predM_v.detach(),
                gtM_v=gtM_v,
                wM_v=wM_v,
                valid_idx_t=valid_idx_t,
                anchor_local_idx_t=anchor_local_idx_t,
            )
            s, R, t = umeyama_similarity(pred_align_v, gt_align_v, w=w_align_v, with_scale=config.with_scale)

        predM_aligned = s * (predM_v @ R.T) + t[None, :]
        diff = predM_aligned - gtM_v
        r = torch.sqrt((diff * diff).sum(dim=1) + 1e-12)

        loss_data = (wM_v * huber(r, delta=config.huber_m)).sum() / (wM_v.sum() + 1e-9)
        loss_reg = config.w_pose_reg * torch.mean((pose - init_pose_ref) ** 2)
        if temporal_pose is not None and config.w_temporal > 0:
            loss_temporal = config.w_temporal * torch.mean((pose - temporal_pose) ** 2)
        else:
            loss_temporal = torch.zeros((), device=device, dtype=torch.float32)
        if temporal_velocity_target is not None and config.w_temporal_velocity > 0:
            loss_temporal_velocity = config.w_temporal_velocity * torch.mean((pose - temporal_velocity_target) ** 2)
        else:
            loss_temporal_velocity = torch.zeros((), device=device, dtype=torch.float32)
        if temporal_pose is not None and temporal_prev_prev_pose is not None and config.w_temporal_accel > 0:
            prev_vel = temporal_pose - temporal_prev_prev_pose
            cur_vel = pose - temporal_pose
            loss_temporal_accel = config.w_temporal_accel * torch.mean((cur_vel - prev_vel) ** 2)
        else:
            loss_temporal_accel = torch.zeros((), device=device, dtype=torch.float32)
        if optimize_hand_pose and config.w_hand_reg > 0:
            loss_hand_reg = config.w_hand_reg * torch.mean((hand - init_hand_ref) ** 2)
        else:
            loss_hand_reg = torch.zeros((), device=device, dtype=torch.float32)
        loss = loss_data + loss_reg + loss_temporal + loss_temporal_velocity + loss_temporal_accel + loss_hand_reg

        loss.backward()
        with torch.no_grad():
            if pose.grad is not None:
                pose.grad[optimize_mask == 0] = 0.0
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        opt.step()

        with torch.no_grad():
            pose.copy_(pose * optimize_mask + frozen_pose_target * (1.0 - optimize_mask))

        loss_hist.append(float(loss.detach().cpu().item()))
        data_loss_hist.append(float(loss_data.detach().cpu().item()))
        if it % 25 == 0 or it == config.iters - 1:
            print(
                f"[{it:04d}] loss={loss_hist[-1]:.6f} "
                f"data={float(loss_data.detach().cpu().item()):.6f} "
                f"temporal={float(loss_temporal.detach().cpu().item()):.6f} "
                f"vel={float(loss_temporal_velocity.detach().cpu().item()):.6f} "
                f"accel={float(loss_temporal_accel.detach().cpu().item()):.6f} "
                f"hand_reg={float(loss_hand_reg.detach().cpu().item()):.6f}"
            )

        cur_loss = loss_hist[-1]
        cur_data_loss = data_loss_hist[-1]
        if cur_loss < (best_loss - improve_tol):
            best_loss = cur_loss
            best_iter = it
            best_pose = pose.detach().clone()
            best_hand = hand.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
        if cur_data_loss < best_data_loss:
            best_data_loss = cur_data_loss
        if (it + 1) >= min_iters and np.isfinite(best_loss) and cur_loss > (best_loss * divergence_ratio):
            print(
                f"[diverge-stop] iter={it:04d} cur_loss={cur_loss:.6f} "
                f"best_loss={best_loss:.6f} ratio={cur_loss / max(best_loss, 1e-12):.2f}"
            )
            break
        if (it + 1) >= min_iters and no_improve >= patience:
            print(f"[early-stop] iter={it:04d} best_loss={best_loss:.6f}")
            break

    final_loss = float(loss_hist[-1]) if loss_hist else float("inf")
    final_data_loss = float(data_loss_hist[-1]) if data_loss_hist else float("inf")
    if best_iter >= 0:
        with torch.no_grad():
            pose.copy_(best_pose)
            pose.copy_(pose * optimize_mask + frozen_pose_target * (1.0 - optimize_mask))
            if optimize_hand_pose:
                hand.copy_(best_hand)

    if config.save_debug_artifacts:
        plot_loss_curve(loss_hist, debug_dir / "loss_curve.png")

    output_sim_scale = 1.0
    output_sim_R = np.eye(3, dtype=np.float32)
    output_sim_t = np.zeros(3, dtype=np.float32)

    with torch.no_grad():
        pose_eff = pose.detach() * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)
        outF = mhr_fk(
            head,
            pose_eff,
            hand.detach(),
            init_scale,
            init_shape,
            init_expr,
            device,
            want_verts=True,
            want_joint=True,
            want_model_params=True,
        )

        verts = apply_repo_camera_flip_xyz(outF[0].squeeze(0))
        keypoints_308 = outF[1].squeeze(0)
        jcoords = apply_repo_camera_flip_xyz(outF[2].squeeze(0))
        model_params = outF[3].squeeze(0)
        jrots = outF[4].squeeze(0)

        k70 = apply_repo_camera_flip_xyz(keypoints_308[:70])
        predM = k70[subset_idx]

        valid_idx_t, wM_masked_t = resolve_valid_indices_for_prediction(
            predM=predM,
            finite_gt_mask_t=finite_gt_mask_t,
            base_wM_t=wM_t,
            min_valid_points=config.min_valid_points,
            strategy=config.zero_weight_strategy,
        )
        predM_v = predM.index_select(0, valid_idx_t)
        gtM_v = gtM_t.index_select(0, valid_idx_t)
        wM_v = wM_masked_t.index_select(0, valid_idx_t)

        pred_align_v, gt_align_v, w_align_v = select_alignment_subset_tensors(
            predM_v=predM_v,
            gtM_v=gtM_v,
            wM_v=wM_v,
            valid_idx_t=valid_idx_t,
            anchor_local_idx_t=anchor_local_idx_t,
        )
        fit_s, fit_R, fit_t = umeyama_similarity(pred_align_v, gt_align_v, w=w_align_v, with_scale=config.with_scale)
        out_s, out_R, out_t = fit_s, fit_R, fit_t
        reused_prev_similarity = False
        if (
            lock_lower_body_pose
            and config.reuse_prev_similarity_when_freeze_lower_body
            and config.init_prev_sim_scale is not None
            and config.init_prev_sim_R is not None
            and config.init_prev_sim_t is not None
        ):
            prev_R = np.asarray(config.init_prev_sim_R, dtype=np.float32).reshape(3, 3)
            prev_t = np.asarray(config.init_prev_sim_t, dtype=np.float32).reshape(3)
            prev_s = float(config.init_prev_sim_scale)
            if np.isfinite(prev_s) and np.isfinite(prev_R).all() and np.isfinite(prev_t).all():
                out_s = to_torch(prev_s, device)
                out_R = to_torch(prev_R, device)
                out_t = to_torch(prev_t, device)
                reused_prev_similarity = True

        k70_aligned = out_s * (k70 @ out_R.T) + out_t[None, :]
        verts_aligned = out_s * (verts @ out_R.T) + out_t[None, :]
        jcoords_aligned = out_s * (jcoords @ out_R.T) + out_t[None, :]
        jrots_aligned = out_R[None, :, :] @ jrots

        predM_aligned_v = fit_s * (predM_v @ fit_R.T) + fit_t[None, :]
        residual_subset = torch.sqrt(((predM_aligned_v - gtM_v) ** 2).sum(dim=1) + 1e-12)
        aligned_data_loss = (wM_v * huber(residual_subset, delta=config.huber_m)).sum() / (wM_v.sum() + 1e-9)
        final_data_loss = float(aligned_data_loss.detach().cpu().item())
        best_data_loss = min(best_data_loss, final_data_loss)
        final_reg_loss = float((config.w_pose_reg * torch.mean((pose - init_pose_ref) ** 2)).detach().cpu().item())
        if temporal_pose is not None and config.w_temporal > 0:
            final_temporal = float((config.w_temporal * torch.mean((pose - temporal_pose) ** 2)).detach().cpu().item())
        else:
            final_temporal = 0.0
        if temporal_velocity_target is not None and config.w_temporal_velocity > 0:
            final_temporal_velocity = float(
                (config.w_temporal_velocity * torch.mean((pose - temporal_velocity_target) ** 2)).detach().cpu().item()
            )
        else:
            final_temporal_velocity = 0.0
        if temporal_pose is not None and temporal_prev_prev_pose is not None and config.w_temporal_accel > 0:
            prev_vel = temporal_pose - temporal_prev_prev_pose
            cur_vel = pose - temporal_pose
            final_temporal_accel = float(
                (config.w_temporal_accel * torch.mean((cur_vel - prev_vel) ** 2)).detach().cpu().item()
            )
        else:
            final_temporal_accel = 0.0
        if optimize_hand_pose and config.w_hand_reg > 0:
            final_hand_reg = float((config.w_hand_reg * torch.mean((hand - init_hand_ref) ** 2)).detach().cpu().item())
        else:
            final_hand_reg = 0.0
        final_loss = final_data_loss + final_reg_loss + final_temporal + final_temporal_velocity + final_temporal_accel + final_hand_reg

        residual_np = residual_subset.cpu().numpy()
        worst = np.argsort(-residual_np)[: config.topk_print]
        print("\nTop residual points (subset indices):")
        for idx in worst:
            subset_pos = int(valid_idx_t[idx].item())
            name = str(subset_names[subset_pos]) if subset_names is not None else f"pt{subset_pos}"
            print(f"  {idx:02d} ({name}): {residual_np[idx]:.4f} m")

        if config.save_debug_artifacts:
            plot_3d_compare(
                gtM[valid_idx_t.cpu().numpy()],
                predM_aligned_v.cpu().numpy(),
                f"Optimized aligned to GT (M={int(valid_idx_t.numel())})",
                debug_dir / "compare_opt_subset.png",
            )

            faces = init_dict.get("faces", None)
            if faces is not None:
                write_ply(debug_dir / "mesh_opt_aligned.ply", verts_aligned.cpu().numpy(), faces)
            else:
                write_ply(debug_dir / "verts_opt_aligned.ply", verts_aligned.cpu().numpy(), None)

            np.savez_compressed(
                debug_dir / "debug_opt.npz",
                best_cam=np.array(best_cam, dtype=object),
                cams=np.array(cams, dtype=object),
                subset_idx=subset_idx,
                gt_subset=gtM,
                w_subset=wM_np,
                init_scores_3d_m=np.array([view_scores_3d[c] for c in cams], dtype=np.float32),
                init_scores_mean_px=np.array([view_mean_px[c] for c in cams], dtype=np.float32),
                final_scale=np.array(float(out_s.cpu().item()), dtype=np.float32),
                final_R=out_R.cpu().numpy(),
                final_t=out_t.cpu().numpy(),
                final_scale_fit=np.array(float(fit_s.cpu().item()), dtype=np.float32),
                final_R_fit=fit_R.cpu().numpy(),
                final_t_fit=fit_t.cpu().numpy(),
                reused_prev_similarity=np.array(int(reused_prev_similarity), dtype=np.int32),
                loss_hist=np.array(loss_hist, dtype=np.float32),
                data_loss_hist=np.array(data_loss_hist, dtype=np.float32),
                best_loss=np.array(best_loss, dtype=np.float32),
                final_loss=np.array(final_loss, dtype=np.float32),
                best_data_loss=np.array(best_data_loss, dtype=np.float32),
                final_data_loss=np.array(final_data_loss, dtype=np.float32),
                best_iter=np.array(best_iter, dtype=np.int32),
                residual_subset_m=np.array(residual_np, dtype=np.float32),
                keep_mask=optimize_mask.cpu().numpy(),
            )

        out_dict = dict(init_dict)
        out_dict["body_pose_params"] = pose_eff.cpu().numpy().astype(np.float32)
        out_dict["hand_pose_params"] = hand.detach().cpu().numpy().astype(np.float32)
        out_dict["pred_keypoints_3d"] = k70_aligned.cpu().numpy().astype(np.float32)
        out_dict["pred_vertices"] = verts_aligned.cpu().numpy().astype(np.float32)
        out_dict["pred_joint_coords"] = jcoords_aligned.cpu().numpy().astype(np.float32)
        out_dict["pred_global_rots"] = jrots_aligned.cpu().numpy().astype(np.float32)
        out_dict["mhr_model_params"] = model_params.cpu().numpy().astype(np.float32)
        out_dict["opt_init_cam"] = best_cam
        out_dict["opt_cam_scores_3d_m"] = view_scores_3d
        out_dict["opt_cam_mean_err_px_refined"] = view_mean_px
        out_dict["opt_sim_scale"] = float(out_s.cpu().item())
        out_dict["opt_sim_R"] = out_R.cpu().numpy()
        out_dict["opt_sim_t"] = out_t.cpu().numpy()
        out_dict["opt_sim_scale_fit"] = float(fit_s.cpu().item())
        out_dict["opt_sim_R_fit"] = fit_R.cpu().numpy()
        out_dict["opt_sim_t_fit"] = fit_t.cpu().numpy()
        out_dict["opt_sim_reused_prev"] = int(reused_prev_similarity)
        out_dict["opt_loss_hist"] = np.array(loss_hist, dtype=np.float32)
        out_dict["opt_data_loss_hist"] = np.array(data_loss_hist, dtype=np.float32)
        out_dict["opt_best_loss"] = float(best_loss)
        out_dict["opt_final_loss"] = float(final_loss)
        out_dict["opt_best_data_loss"] = float(best_data_loss)
        out_dict["opt_final_data_loss"] = float(final_data_loss)
        out_dict["opt_best_iter"] = int(best_iter)
        growth = safe_growth(final_loss, best_loss)
        data_growth = safe_growth(final_data_loss, best_data_loss)
        out_dict["opt_loss_growth_ratio"] = growth
        out_dict["opt_data_loss_growth_ratio"] = data_growth
        out_dict["opt_used_temporal_init"] = int(used_temporal_init)
        out_dict["opt_temporal_weight"] = float(config.w_temporal)
        out_dict["opt_temporal_velocity_weight"] = float(config.w_temporal_velocity)
        out_dict["opt_temporal_accel_weight"] = float(config.w_temporal_accel)
        out_dict["opt_hand_reg_weight"] = float(config.w_hand_reg)
        out_dict["opt_optimize_hand_pose"] = int(bool(optimize_hand_pose))
        out_dict["opt_use_anchor_similarity"] = int(bool(config.use_anchor_similarity))
        out_dict["opt_temporal_extrapolation"] = float(config.temporal_extrapolation)
        out_dict["opt_subset_indices"] = subset_idx
        out_dict["opt_points3d_refined"] = gtM
        out_dict["opt_fixed_hand_pose_params"] = int(config.fixed_hand_pose_params is not None)
        out_dict["opt_fixed_scale_params"] = int(config.fixed_scale_params is not None)
        out_dict["opt_fixed_shape_params"] = int(config.fixed_shape_params is not None)
        out_dict["opt_fixed_expr_params"] = int(config.fixed_expr_params is not None)
        out_dict["opt_fixed_lower_body_pose_params"] = int(use_fixed_lower_body_pose)

        is_bad_loss = classify_bad_optimization(
            config=config,
            best_loss=best_loss,
            final_loss=final_loss,
            best_data_loss=best_data_loss,
            final_data_loss=final_data_loss,
        )
        out_dict["opt_is_bad_loss"] = int(is_bad_loss)

        validate_optimization_result_dict(
            out_dict,
            require_geometry=True,
            require_mhr_compact=True,
        )
        np.save(out_npy, out_dict, allow_pickle=True)
        output_sim_scale = float(out_s.cpu().item())
        output_sim_R = out_R.cpu().numpy().astype(np.float32)
        output_sim_t = out_t.cpu().numpy().astype(np.float32)

    print(f"\nSaved optimized npy: {out_npy}")
    print(
        f"[quality] best_loss={best_loss:.6f} final_loss={final_loss:.6f} "
        f"best_data={best_data_loss:.6f} final_data={final_data_loss:.6f} "
        f"best_iter={best_iter} temporal_init={int(used_temporal_init)}"
    )
    if config.save_debug_artifacts:
        print(f"Saved debug dir: {debug_dir}")
    else:
        print("Debug artifacts disabled (--no_debug_artifacts).")

    is_bad_loss = classify_bad_optimization(
        config=config,
        best_loss=best_loss,
        final_loss=final_loss,
        best_data_loss=best_data_loss,
        final_data_loss=final_data_loss,
    )
    return OptimizationRunResult(
        out_npy=out_npy,
        debug_dir=debug_dir,
        best_cam=best_cam,
        loss_history=loss_hist,
        best_loss=float(best_loss),
        final_loss=float(final_loss),
        best_data_loss=float(best_data_loss),
        final_data_loss=float(final_data_loss),
        best_iter=int(best_iter),
        used_temporal_init=bool(used_temporal_init),
        is_bad_loss=is_bad_loss,
        best_pose=(pose.detach() * optimize_mask + frozen_pose_target * (1.0 - optimize_mask)).cpu().numpy().astype(np.float32),
        sim_scale=output_sim_scale,
        sim_R=output_sim_R,
        sim_t=output_sim_t,
    )
