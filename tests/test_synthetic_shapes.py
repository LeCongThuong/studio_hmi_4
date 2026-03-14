from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.LINE_AA = 16
    cv2.FONT_HERSHEY_SIMPLEX = 0

    def _fake_imwrite(path, img):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fake-image")
        return True

    def _fake_imread(path, flags=None):
        if not Path(path).exists():
            return None
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def _fake_rodrigues(src):
        arr = np.asarray(src, dtype=np.float64)
        if arr.shape == (3, 3):
            return np.zeros((3, 1), dtype=np.float64), None
        return np.eye(3, dtype=np.float64), None

    def _fake_project_points(points, rvec, tvec, K, D):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        t = np.asarray(tvec, dtype=np.float64).reshape(1, 3)
        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
        cam = pts + t
        uv = np.empty((pts.shape[0], 2), dtype=np.float64)
        uv[:, 0] = K_arr[0, 0] * (cam[:, 0] / cam[:, 2]) + K_arr[0, 2]
        uv[:, 1] = K_arr[1, 1] * (cam[:, 1] / cam[:, 2]) + K_arr[1, 2]
        return uv.reshape(-1, 1, 2), None

    def _fake_undistort_points(pts, K, D):
        pts_arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
        out = np.empty_like(pts_arr)
        out[:, 0] = (pts_arr[:, 0] - K_arr[0, 2]) / K_arr[0, 0]
        out[:, 1] = (pts_arr[:, 1] - K_arr[1, 2]) / K_arr[1, 1]
        return out.reshape(-1, 1, 2)

    def _fake_passthrough(img, *args, **kwargs):
        return img

    cv2.imwrite = _fake_imwrite
    cv2.imread = _fake_imread
    cv2.Rodrigues = _fake_rodrigues
    cv2.projectPoints = _fake_project_points
    cv2.undistortPoints = _fake_undistort_points
    cv2.line = _fake_passthrough
    cv2.circle = _fake_passthrough
    cv2.rectangle = _fake_passthrough
    cv2.putText = _fake_passthrough
    sys.modules["cv2"] = cv2

from studio_hmi_4.common import (
    load_npy_dict,
    validate_optimization_result_dict,
    validate_stage1_prediction_dict,
    validate_triangulation_bundle,
)
from studio_hmi_4.export import compact as compact_export
from studio_hmi_4.sequence.recovery import recover_missing_and_bad_frames
from studio_hmi_4.sequence.runner import FullPipelineConfig, run_full_pipeline
from studio_hmi_4.sequence.types import FramePipelineResult
from studio_hmi_4.stage1.runner import Demo2Config, Demo2RunResult, FrameResult, run_demo
from studio_hmi_4.stage2.runner import MHRSubsetSelector, TriangulationConfig, run_triangulation
from studio_hmi_4.stage3.runner import (
    ALIGNMENT_ANCHOR_NAMES,
    OptimizationConfig,
    OptimizationRunResult,
    OptimizationRuntime,
    SUBSET_LOSS_WEIGHT_BY_NAME,
    apply_repo_camera_flip_xyz,
    build_alignment_anchor_local_indices,
    build_subset_loss_weights,
    mhr_fk,
    resolve_valid_indices_for_prediction,
    resolve_lower_body_pose_indices,
    run_optimization,
    sanitize_subset_and_weights,
)


def _make_stage1_prediction(seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "pred_keypoints_2d": rng.normal(size=(70, 2)).astype(np.float32),
        "body_pose_params": rng.normal(scale=0.01, size=(133,)).astype(np.float32),
        "hand_pose_params": rng.normal(scale=0.01, size=(108,)).astype(np.float32),
        "scale_params": rng.normal(scale=0.01, size=(28,)).astype(np.float32),
        "shape_params": rng.normal(scale=0.01, size=(45,)).astype(np.float32),
        "expr_params": rng.normal(scale=0.01, size=(72,)).astype(np.float32),
        "pred_keypoints_3d": rng.normal(scale=0.01, size=(70, 3)).astype(np.float32),
        "pred_vertices": rng.normal(scale=0.01, size=(10, 3)).astype(np.float32),
        "pred_joint_coords": rng.normal(scale=0.01, size=(70, 3)).astype(np.float32),
        "pred_global_rots": np.tile(np.eye(3, dtype=np.float32), (70, 1, 1)),
        "mhr_model_params": rng.normal(scale=0.01, size=(204,)).astype(np.float32),
        "faces": np.array([[0, 1, 2]], dtype=np.int32),
    }


def _write_blank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Failed to write image: {path}")


class _FakeEstimator:
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    def process_one_image(self, image_path: str, bbox_thr: float, use_mask: bool):
        pred = _make_stage1_prediction(seed=sum(map(ord, Path(image_path).name)))
        return [pred]


def _subset() -> tuple[np.ndarray, np.ndarray]:
    selector = MHRSubsetSelector("mhr70.py")
    subset = selector.build_subset()
    return subset.subset_indices, subset.subset_names


def _camera_block(tx: float) -> dict[str, object]:
    return {
        "matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "translation": [tx, 0.0, 0.0],
        "size": [640, 480],
    }


def _write_simple_toml(path: Path, cams: list[str]) -> None:
    tx_map = {"left": -0.15, "front": 0.0, "right": 0.15}
    chunks = []
    for cam in cams:
        block = _camera_block(tx_map[cam])
        chunks.append(f"[{cam}]")
        chunks.append(f"matrix = {json.dumps(block['matrix'])}")
        chunks.append(f"distortions = {json.dumps(block['distortions'])}")
        chunks.append(f"rotation = {json.dumps(block['rotation'])}")
        chunks.append(f"translation = {json.dumps(block['translation'])}")
        chunks.append(f"size = {json.dumps(block['size'])}")
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")


def _project_points(points3d: np.ndarray, tx: float) -> np.ndarray:
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[tx], [0.0], [0.0]], dtype=np.float64)
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    uv, _ = cv2.projectPoints(points3d.reshape(-1, 1, 3), rvec, tvec, K, D)
    return uv.reshape(-1, 2).astype(np.float32)


class _FakeHead:
    def eval(self):
        return self

    def mhr_forward(
        self,
        *,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params,
        return_keypoints,
        return_joint_coords,
        return_model_params,
        return_joint_rotations,
    ):
        batch = body_pose_params.shape[0]
        pose = body_pose_params.to(torch.float32)
        hand = hand_pose_params.to(torch.float32)
        scale = scale_params.to(torch.float32)
        shape = shape_params.to(torch.float32)
        expr = expr_params.to(torch.float32)

        flat = torch.cat([pose, hand, scale, shape, expr], dim=1)
        need = 308 * 3
        if flat.shape[1] < need:
            flat = torch.nn.functional.pad(flat, (0, need - flat.shape[1]))
        keypoints = flat[:, :need].reshape(batch, 308, 3) * 0.05

        verts_flat = flat
        if verts_flat.shape[1] < 30:
            verts_flat = torch.nn.functional.pad(verts_flat, (0, 30 - verts_flat.shape[1]))
        verts = verts_flat[:, :30].reshape(batch, 10, 3) * 0.05

        model_params = torch.zeros((batch, 204), device=pose.device, dtype=pose.dtype)
        pose_copy = min(pose.shape[1], 130)
        model_params[:, 6 : 6 + pose_copy] = pose[:, :pose_copy]
        scale_copy = min(scale.shape[1], 68)
        model_params[:, 136 : 136 + scale_copy] = scale[:, :scale_copy]
        joint_rots = (
            torch.eye(3, device=pose.device, dtype=pose.dtype)
            .view(1, 1, 3, 3)
            .repeat(batch, 308, 1, 1)
        )
        return verts, keypoints, keypoints.clone(), model_params, joint_rots


def _build_fake_runtime() -> OptimizationRuntime:
    return OptimizationRuntime(
        device=torch.device("cpu"),
        head=_FakeHead(),
        hand_mask=torch.zeros(133, dtype=torch.bool),
        keep_mask=torch.ones(133, dtype=torch.float32),
    )


class SyntheticShapeTests(unittest.TestCase):
    def test_lower_body_pose_index_table_matches_verified_subset(self):
        expected_133 = np.array(
            [44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 58, 116, 117, 118, 120, 121, 122, 128, 129],
            dtype=np.int64,
        )
        expected_204 = np.array(
            sorted(set(list(range(0, 6)) + [6 + int(i) for i in expected_133])),
            dtype=np.int64,
        )
        np.testing.assert_array_equal(resolve_lower_body_pose_indices(133), expected_133)
        np.testing.assert_array_equal(resolve_lower_body_pose_indices(204), expected_204)

    def test_stage2_subset_includes_torso_alignment_points(self):
        _subset_idx, subset_names = _subset()
        subset_name_set = {str(name) for name in subset_names.tolist()}
        self.assertIn("neck", subset_name_set)
        self.assertIn("left_acromion", subset_name_set)
        self.assertIn("right_acromion", subset_name_set)
        self.assertIn("left_elbow", subset_name_set)
        self.assertIn("right_elbow", subset_name_set)
        self.assertIn("left_wrist", subset_name_set)
        self.assertIn("right_wrist", subset_name_set)
        self.assertIn("left_hip", subset_name_set)
        self.assertIn("right_hip", subset_name_set)

    def test_stage3_alignment_anchors_are_torso_only(self):
        self.assertEqual(
            ALIGNMENT_ANCHOR_NAMES,
            {"left_hip", "right_hip", "neck", "left_acromion", "right_acromion"},
        )
        _subset_idx, subset_names = _subset()
        local_idx = build_alignment_anchor_local_indices(subset_names)
        local_names = [str(subset_names[idx]) for idx in local_idx.tolist()]
        self.assertEqual(
            local_names,
            ["left_hip", "right_hip", "neck", "left_acromion", "right_acromion"],
        )

    def test_stage3_loss_weights_downweight_torso_and_shoulders(self):
        _subset_idx, subset_names = _subset()
        weights = build_subset_loss_weights(subset_names)
        mapping = {
            str(name): float(weights[idx])
            for idx, name in enumerate(subset_names.tolist())
        }

        for name, value in SUBSET_LOSS_WEIGHT_BY_NAME.items():
            self.assertAlmostEqual(mapping[name], value, places=6)
        for name, value in mapping.items():
            if name not in SUBSET_LOSS_WEIGHT_BY_NAME:
                self.assertEqual(value, 1.0)

    def test_stage3_uniform_finite_fallback_respects_allowed_mask(self):
        gtM = np.zeros((5, 3), dtype=np.float32)
        wM = np.zeros((5,), dtype=np.float32)
        allowed_mask = np.array([1, 1, 1, 0, 0], dtype=bool)

        finite_gt_mask, sanitized_wM, _ = sanitize_subset_and_weights(
            gtM=gtM,
            wM=wM,
            min_valid_points=3,
            strategy="uniform_finite",
            allowed_mask=allowed_mask,
        )
        np.testing.assert_array_equal(sanitized_wM, np.array([1, 1, 1, 0, 0], dtype=np.float32))

        valid_idx_t, masked_wM_t = resolve_valid_indices_for_prediction(
            predM=torch.zeros((5, 3), dtype=torch.float32),
            finite_gt_mask_t=torch.from_numpy(finite_gt_mask),
            base_wM_t=torch.zeros((5,), dtype=torch.float32),
            min_valid_points=3,
            strategy="uniform_finite",
            allowed_mask_t=torch.from_numpy(allowed_mask),
        )
        self.assertEqual(valid_idx_t.tolist(), [0, 1, 2])
        np.testing.assert_array_equal(masked_wM_t.cpu().numpy(), np.array([1, 1, 1, 0, 0], dtype=np.float32))

    def test_root_wrappers_import_from_copied_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            copy_root = tmp_path / "sam3d_body_root"
            copy_root.mkdir(parents=True, exist_ok=True)

            shutil.copytree(Path("studio_hmi_4"), copy_root / "studio_hmi_4")
            shutil.copy2(Path("mhr70.py"), copy_root / "mhr70.py")
            for name in (
                "sam3d_inference.py",
                "triangulate_mhr3d_gt.py",
                "optimize_mhr_pose.py",
                "run_full_pipeline.py",
                "video_temporal_utils.py",
                "export_mhr_final_results.py",
                "run_mhr_repo_from_export.py",
                "run_wilor_precompute.py",
                "extract_per_tick_frames_from_csv.py",
                "interactive_mesh_video_viewer.py",
            ):
                shutil.copy2(Path(name), copy_root / name)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import run_full_pipeline, sam3d_inference, "
                        "triangulate_mhr3d_gt, optimize_mhr_pose, "
                        "interactive_mesh_video_viewer; "
                        "print('copy-root-ok')"
                    ),
                ],
                cwd=copy_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"wrapper import failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
            self.assertIn("copy-root-ok", proc.stdout)

    def test_stage1_run_demo_shape_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_root = tmp_path / "images"
            _write_blank_image(image_root / "0" / "front.jpg")
            _write_blank_image(image_root / "0" / "left.jpg")

            config = Demo2Config(
                image_folder=str(image_root),
                checkpoint_path="dummy.ckpt",
                output_folder=str(tmp_path / "stage1"),
            )
            result = run_demo(config, estimator=_FakeEstimator(), show_progress=False)

            self.assertIsInstance(result, Demo2RunResult)
            saved = load_npy_dict(result.frames[0].npy_path)
            contract = validate_stage1_prediction_dict(saved, require_pose_blocks=True)
            self.assertEqual(contract.pred_keypoints_2d.shape, (70, 2))
            self.assertEqual(contract.body_pose_params.shape, (133,))

    def test_stage2_triangulation_with_synthetic_views(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cams = ["left", "front", "right"]
            npy_dir = tmp_path / "npy"
            npy_dir.mkdir()
            toml_path = tmp_path / "cams.toml"
            _write_simple_toml(toml_path, cams)

            rng = np.random.default_rng(7)
            points3d = rng.uniform(
                low=[-0.2, -0.2, 3.0],
                high=[0.2, 0.2, 4.0],
                size=(70, 3),
            ).astype(np.float64)
            tx_map = {"left": -0.15, "front": 0.0, "right": 0.15}
            for cam in cams:
                pred = _make_stage1_prediction(seed=len(cam))
                pred["pred_keypoints_2d"] = _project_points(points3d, tx=tx_map[cam])
                np.save(npy_dir / f"{cam}.npy", pred, allow_pickle=True)

            out_npz = tmp_path / "triangulated.npz"
            config = TriangulationConfig(
                mhr_py="mhr70.py",
                caliscope_toml=str(toml_path),
                cams=cams,
                npy_dir=str(npy_dir),
                out_npz=str(out_npz),
            )
            result = run_triangulation(config)
            self.assertTrue(result.out_npz.exists())

            z = np.load(result.out_npz, allow_pickle=True)
            contract = validate_triangulation_bundle(z)
            self.assertEqual(contract.points3d_refined.shape[1], 3)
            self.assertEqual(contract.subset_indices.shape[0], 51)

    def test_stage3_optimization_with_fake_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cams = ["left", "front", "right"]
            npy_dir = tmp_path / "stage1_npy"
            npy_dir.mkdir()
            subset_idx, subset_names = _subset()
            runtime = _build_fake_runtime()

            init_pred = _make_stage1_prediction(seed=11)
            for cam in cams:
                np.save(npy_dir / f"{cam}.npy", init_pred, allow_pickle=True)

            device = runtime.device
            pose = torch.from_numpy(np.asarray(init_pred["body_pose_params"], dtype=np.float32)).to(device)
            hand = torch.from_numpy(np.asarray(init_pred["hand_pose_params"], dtype=np.float32)).to(device)
            scale = torch.from_numpy(np.asarray(init_pred["scale_params"], dtype=np.float32)).to(device)
            shape = torch.from_numpy(np.asarray(init_pred["shape_params"], dtype=np.float32)).to(device)
            expr = torch.from_numpy(np.asarray(init_pred["expr_params"], dtype=np.float32)).to(device)
            fk_out = mhr_fk(runtime.head, pose, hand, scale, shape, expr, device, want_verts=True, want_joint=True, want_model_params=True)
            k70 = apply_repo_camera_flip_xyz(fk_out[1].squeeze(0)[:70]).cpu().numpy()
            gt_subset = k70[subset_idx]

            npz_path = tmp_path / "triangulated.npz"
            np.savez_compressed(
                npz_path,
                subset_indices=subset_idx,
                subset_names=subset_names,
                points3d_refined=gt_subset.astype(np.float32),
                inlier_mask=np.ones((subset_idx.shape[0], len(cams)), dtype=np.uint8),
                left_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                front_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                right_mean_err_px_refined=np.array(0.1, dtype=np.float32),
            )

            out_npy = tmp_path / "opt_out.npy"
            config = OptimizationConfig(
                npz=npz_path,
                npy_dir=npy_dir,
                cams=cams,
                out_npy=out_npy,
                hf_repo="fake/repo",
                device="cpu",
                iters=5,
                save_debug_artifacts=False,
            )
            result = run_optimization(config, runtime=runtime)
            self.assertIsInstance(result, OptimizationRunResult)
            saved = load_npy_dict(out_npy)
            contract = validate_optimization_result_dict(
                saved,
                require_geometry=True,
                require_mhr_compact=True,
            )
            self.assertEqual(contract.body_pose_params.shape, (133,))
            self.assertEqual(contract.pred_keypoints_3d.shape, (70, 3))

    def test_stage3_optimization_can_fix_lower_body_from_reference_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cams = ["left", "front", "right"]
            npy_dir = tmp_path / "stage1_npy"
            npy_dir.mkdir()
            subset_idx, subset_names = _subset()
            runtime = _build_fake_runtime()

            init_pred = _make_stage1_prediction(seed=21)
            for cam in cams:
                np.save(npy_dir / f"{cam}.npy", init_pred, allow_pickle=True)

            device = runtime.device
            pose = torch.from_numpy(np.asarray(init_pred["body_pose_params"], dtype=np.float32)).to(device)
            hand = torch.from_numpy(np.asarray(init_pred["hand_pose_params"], dtype=np.float32)).to(device)
            scale = torch.from_numpy(np.asarray(init_pred["scale_params"], dtype=np.float32)).to(device)
            shape = torch.from_numpy(np.asarray(init_pred["shape_params"], dtype=np.float32)).to(device)
            expr = torch.from_numpy(np.asarray(init_pred["expr_params"], dtype=np.float32)).to(device)
            fk_out = mhr_fk(runtime.head, pose, hand, scale, shape, expr, device, want_verts=True, want_joint=True, want_model_params=True)
            k70 = apply_repo_camera_flip_xyz(fk_out[1].squeeze(0)[:70]).cpu().numpy()
            gt_subset = k70[subset_idx]

            npz_path = tmp_path / "triangulated.npz"
            np.savez_compressed(
                npz_path,
                subset_indices=subset_idx,
                subset_names=subset_names,
                points3d_refined=gt_subset.astype(np.float32),
                inlier_mask=np.ones((subset_idx.shape[0], len(cams)), dtype=np.uint8),
                left_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                front_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                right_mean_err_px_refined=np.array(0.1, dtype=np.float32),
            )

            ref_pose = np.asarray(init_pred["body_pose_params"], dtype=np.float32).copy()
            lower_body_idxs = resolve_lower_body_pose_indices(ref_pose.shape[0])
            ref_pose[lower_body_idxs] = np.linspace(-0.5, 0.5, lower_body_idxs.shape[0], dtype=np.float32)

            out_npy = tmp_path / "opt_out_fixed_lower.npy"
            config = OptimizationConfig(
                npz=npz_path,
                npy_dir=npy_dir,
                cams=cams,
                out_npy=out_npy,
                hf_repo="fake/repo",
                device="cpu",
                iters=5,
                save_debug_artifacts=False,
                fixed_lower_body_pose_params=ref_pose,
            )
            result = run_optimization(config, runtime=runtime)

            self.assertIsInstance(result, OptimizationRunResult)
            np.testing.assert_allclose(result.best_pose[lower_body_idxs], ref_pose[lower_body_idxs], atol=1e-6, rtol=0.0)

    def test_full_pipeline_orchestrator_with_synthetic_stage_hooks(self):
        from studio_hmi_4.sequence import runner as sequence_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_root = tmp_path / "frames"
            for frame_idx in range(3):
                for cam in ("left", "front"):
                    _write_blank_image(image_root / str(frame_idx) / f"{cam}.jpg")

            subset_idx, subset_names = _subset()

            def fake_run_demo(config):
                output_root = Path(config.output_folder)
                npy_root = output_root / "npy"
                frames = []
                for image_path in sorted(image_root.rglob("*.jpg")):
                    rel_dir = image_path.parent.relative_to(image_root).as_posix()
                    cam = image_path.stem
                    pred = _make_stage1_prediction(seed=int(rel_dir) * 10 + len(cam))
                    out_dir = npy_root / rel_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{cam}.npy"
                    np.save(out_path, pred, allow_pickle=True)
                    frames.append(
                        FrameResult(
                            image_path=image_path,
                            rel_dir=rel_dir,
                            npy_path=out_path,
                            mhr_params_path=None,
                            has_prediction=True,
                        )
                    )
                (output_root / "stage1_meta.json").write_text("{}", encoding="utf-8")
                return Demo2RunResult(
                    output_root=output_root,
                    npy_root=npy_root,
                    render_root=output_root / "render",
                    mesh_root=output_root / "mesh",
                    mhr_params_root=None,
                    frames=frames,
                )

            def fake_run_triangulation(config):
                out_npz = Path(config.out_npz)
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_npz,
                    subset_indices=subset_idx,
                    subset_names=subset_names,
                    points3d_refined=np.ones((subset_idx.shape[0], 3), dtype=np.float32) * float(out_npz.parent.name or 0),
                    inlier_mask=np.ones((subset_idx.shape[0], len(config.cams)), dtype=np.uint8),
                    left_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                    front_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                )
                return None

            def fake_build_runtime(config):
                return object()

            def fake_run_optimization(config, runtime):
                frame_idx = int(Path(config.out_npy).parent.name)
                out_path = Path(config.out_npy)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_dict = _make_stage1_prediction(seed=frame_idx)
                out_dict["pred_keypoints_3d"] = np.full((70, 3), frame_idx, dtype=np.float32)
                out_dict["pred_vertices"] = np.full((10, 3), frame_idx, dtype=np.float32)
                out_dict["pred_joint_coords"] = np.full((70, 3), frame_idx, dtype=np.float32)
                out_dict["pred_global_rots"] = np.tile(np.eye(3, dtype=np.float32), (70, 1, 1))
                out_dict["opt_is_bad_loss"] = int(frame_idx == 1)
                out_dict["opt_best_loss"] = 1e-6
                out_dict["opt_final_loss"] = 1e-6
                out_dict["opt_best_data_loss"] = 1e-6
                out_dict["opt_final_data_loss"] = 1e-6
                out_dict["opt_best_iter"] = 1
                out_dict["opt_sim_scale"] = 1.0
                out_dict["opt_sim_R"] = np.eye(3, dtype=np.float32)
                out_dict["opt_sim_t"] = np.zeros(3, dtype=np.float32)
                np.save(out_path, out_dict, allow_pickle=True)
                return OptimizationRunResult(
                    out_npy=out_path,
                    debug_dir=out_path.parent / "debug_opt",
                    best_cam="front",
                    loss_history=[1e-6],
                    best_loss=1e-6,
                    final_loss=1e-6,
                    best_data_loss=1e-6,
                    final_data_loss=1e-6,
                    best_iter=1,
                    used_temporal_init=frame_idx > 0,
                    is_bad_loss=frame_idx == 1,
                    best_pose=np.zeros((133,), dtype=np.float32),
                    sim_scale=1.0,
                    sim_R=np.eye(3, dtype=np.float32),
                    sim_t=np.zeros((3,), dtype=np.float32),
                )

            with mock.patch.object(sequence_runner, "run_demo", side_effect=fake_run_demo), \
                mock.patch.object(sequence_runner, "run_triangulation", side_effect=fake_run_triangulation), \
                mock.patch.object(sequence_runner, "build_optimization_runtime", side_effect=fake_build_runtime), \
                mock.patch.object(sequence_runner, "run_optimization", side_effect=fake_run_optimization):
                config = FullPipelineConfig(
                    image_folder=str(image_root),
                    output_root=str(tmp_path / "pipeline_out"),
                    cams=["left", "front"],
                    caliscope_toml=str(tmp_path / "unused.toml"),
                    checkpoint_path="dummy.ckpt",
                    mhr_path="dummy.pt",
                    hf_repo="fake/repo",
                    device="cpu",
                    min_views=2,
                    save_sequence_mp4=False,
                )
                result = run_full_pipeline(config)

            self.assertTrue(any(frame.status == "recovered_interpolated" for frame in result.frames))
            self.assertIsNotNone(result.summary_json)
            self.assertTrue(result.summary_json.exists())
            smoothed = [frame.smoothed_npy for frame in result.frames if frame.smoothed_npy is not None]
            self.assertEqual(len(smoothed), len(result.frames))

    def test_compact_export_without_forward_uses_shape_valid_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            optimization_root = tmp_path / "optimization"
            frame_dir = optimization_root / "0"
            frame_dir.mkdir(parents=True, exist_ok=True)
            out_dict = _make_stage1_prediction(seed=123)
            out_dict["opt_is_bad_loss"] = 0
            out_dict["mhr_model_params"] = np.linspace(0.0, 1.0, 204, dtype=np.float32)
            np.save(frame_dir / "opt_out_smoothed.npy", out_dict, allow_pickle=True)

            args = argparse.Namespace(
                optimization_root=str(optimization_root),
                output_root=str(tmp_path / "export"),
                npy_name="opt_out_smoothed.npy",
                fallback_npy_name="opt_out.npy",
                overwrite=False,
                max_frames=0,
                skip_bad=True,
                no_decomposed_params=False,
                no_run_mhr_forward=True,
                hf_repo=None,
                ckpt=None,
                mhr_pt="",
                device="cpu",
                debug_vis=False,
                mesh_name="mesh.ply",
                quiet=False,
            )
            rc = compact_export.main(args)
            self.assertEqual(rc, 0)
            self.assertTrue((tmp_path / "export" / "0" / "mhr_params.npy").exists())
            self.assertTrue((tmp_path / "export" / "0" / "mhr_params.npz").exists())
            self.assertTrue((tmp_path / "export" / "0" / "mesh.ply").exists())

    def test_full_pipeline_can_skip_inference_and_triangulation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from studio_hmi_4.sequence import runner as sequence_runner

            tmp_path = Path(tmp_dir)
            image_root = tmp_path / "images"
            output_root = tmp_path / "pipeline_out"
            inference_npy_root = output_root / "inference" / "npy"
            triangulation_root = output_root / "triangulation"
            cams = ["left", "front"]
            subset_idx, subset_names = _subset()
            reference_front_pose = None

            for frame_idx in range(2):
                frame_rel = str(frame_idx)
                frame_dir = image_root / frame_rel
                inference_frame_dir = inference_npy_root / frame_rel
                inference_frame_dir.mkdir(parents=True, exist_ok=True)
                for cam in cams:
                    _write_blank_image(frame_dir / f"{cam}.png")
                    pred = _make_stage1_prediction(seed=frame_idx * 10 + len(cam))
                    np.save(inference_frame_dir / f"{cam}.npy", pred, allow_pickle=True)
                    if frame_idx == 0 and cam == "front":
                        reference_front_pose = np.asarray(pred["body_pose_params"], dtype=np.float32).copy()
                tri_out = triangulation_root / frame_rel / "triangulated.npz"
                tri_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    tri_out,
                    subset_indices=subset_idx,
                    subset_names=subset_names,
                    points3d_refined=np.ones((subset_idx.shape[0], 3), dtype=np.float32) * float(frame_idx),
                    inlier_mask=np.ones((subset_idx.shape[0], len(cams)), dtype=np.uint8),
                    left_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                    front_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                )

            build_calls = []
            opt_calls = []

            def fail_run_demo(config):
                raise AssertionError("run_demo should not be called when --skip_inference is enabled")

            def fail_run_triangulation(config):
                raise AssertionError("run_triangulation should not be called when --skip_triangulation is enabled")

            def fake_build_runtime(config):
                build_calls.append(Path(config.npy_dir))
                return object()

            def fake_run_optimization(config, runtime):
                frame_idx = int(Path(config.out_npy).parent.name)
                opt_calls.append(frame_idx)
                self.assertIsNotNone(reference_front_pose)
                self.assertIsNotNone(config.fixed_lower_body_pose_params)
                np.testing.assert_allclose(config.fixed_lower_body_pose_params, reference_front_pose, atol=1e-6, rtol=0.0)
                self.assertFalse(config.freeze_lower_body)
                out_path = Path(config.out_npy)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_dict = _make_stage1_prediction(seed=frame_idx)
                out_dict["pred_keypoints_3d"] = np.full((70, 3), frame_idx, dtype=np.float32)
                out_dict["pred_vertices"] = np.full((10, 3), frame_idx, dtype=np.float32)
                out_dict["pred_joint_coords"] = np.full((70, 3), frame_idx, dtype=np.float32)
                out_dict["pred_global_rots"] = np.tile(np.eye(3, dtype=np.float32), (70, 1, 1))
                out_dict["opt_is_bad_loss"] = 0
                out_dict["opt_best_loss"] = 1e-6
                out_dict["opt_final_loss"] = 1e-6
                out_dict["opt_best_data_loss"] = 1e-6
                out_dict["opt_final_data_loss"] = 1e-6
                out_dict["opt_best_iter"] = 1
                out_dict["opt_sim_scale"] = 1.0
                out_dict["opt_sim_R"] = np.eye(3, dtype=np.float32)
                out_dict["opt_sim_t"] = np.zeros(3, dtype=np.float32)
                np.save(out_path, out_dict, allow_pickle=True)
                return OptimizationRunResult(
                    out_npy=out_path,
                    debug_dir=out_path.parent / "debug_opt",
                    best_cam="front",
                    loss_history=[1e-6],
                    best_loss=1e-6,
                    final_loss=1e-6,
                    best_data_loss=1e-6,
                    final_data_loss=1e-6,
                    best_iter=1,
                    used_temporal_init=False,
                    is_bad_loss=False,
                    best_pose=np.zeros((133,), dtype=np.float32),
                    sim_scale=1.0,
                    sim_R=np.eye(3, dtype=np.float32),
                    sim_t=np.zeros((3,), dtype=np.float32),
                )

            with mock.patch.object(sequence_runner, "run_demo", side_effect=fail_run_demo), \
                mock.patch.object(sequence_runner, "run_triangulation", side_effect=fail_run_triangulation), \
                mock.patch.object(sequence_runner, "build_optimization_runtime", side_effect=fake_build_runtime), \
                mock.patch.object(sequence_runner, "run_optimization", side_effect=fake_run_optimization):
                config = FullPipelineConfig(
                    image_folder=str(image_root),
                    output_root=str(output_root),
                    cams=cams,
                    caliscope_toml=str(tmp_path / "unused.toml"),
                    checkpoint_path="dummy.ckpt",
                    mhr_path="dummy.pt",
                    hf_repo="fake/repo",
                    device="cpu",
                    min_views=2,
                    skip_inference=True,
                    skip_triangulation=True,
                    fixed_lower_body_pose_frame_idx=0,
                    fixed_lower_body_pose_cam="front",
                    freeze_lower_body=True,
                    overwrite=True,
                    save_sequence_mp4=False,
                )
                result = run_full_pipeline(config)

            self.assertEqual(len(build_calls), 1)
            self.assertEqual(sorted(opt_calls), [0, 1])
            self.assertTrue(all(frame.status == "ok" for frame in result.frames))
            self.assertTrue(all(frame.optimized_npy is not None and frame.optimized_npy.exists() for frame in result.frames))

    def test_full_pipeline_fixed_mhr_params_only_fix_scale_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from studio_hmi_4.sequence import runner as sequence_runner

            tmp_path = Path(tmp_dir)
            image_root = tmp_path / "images"
            output_root = tmp_path / "pipeline_out"
            inference_npy_root = output_root / "inference" / "npy"
            triangulation_root = output_root / "triangulation"
            cams = ["left", "front"]
            subset_idx, subset_names = _subset()
            reference_front_scale = None
            reference_front_shape = None
            reference_front_expr = None

            for frame_idx in range(2):
                frame_rel = str(frame_idx)
                frame_dir = image_root / frame_rel
                inference_frame_dir = inference_npy_root / frame_rel
                inference_frame_dir.mkdir(parents=True, exist_ok=True)
                for cam in cams:
                    _write_blank_image(frame_dir / f"{cam}.png")
                    pred = _make_stage1_prediction(seed=frame_idx * 10 + len(cam))
                    np.save(inference_frame_dir / f"{cam}.npy", pred, allow_pickle=True)
                    if frame_idx == 0 and cam == "front":
                        reference_front_scale = np.asarray(pred["scale_params"], dtype=np.float32).copy()
                        reference_front_shape = np.asarray(pred["shape_params"], dtype=np.float32).copy()
                        reference_front_expr = np.asarray(pred["expr_params"], dtype=np.float32).copy()
                tri_out = triangulation_root / frame_rel / "triangulated.npz"
                tri_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    tri_out,
                    subset_indices=subset_idx,
                    subset_names=subset_names,
                    points3d_refined=np.ones((subset_idx.shape[0], 3), dtype=np.float32) * float(frame_idx),
                    inlier_mask=np.ones((subset_idx.shape[0], len(cams)), dtype=np.uint8),
                    left_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                    front_mean_err_px_refined=np.array(0.1, dtype=np.float32),
                )

            def fail_run_demo(config):
                raise AssertionError("run_demo should not be called when --skip_inference is enabled")

            def fail_run_triangulation(config):
                raise AssertionError("run_triangulation should not be called when --skip_triangulation is enabled")

            def fake_build_runtime(config):
                return object()

            def fake_run_optimization(config, runtime):
                self.assertIsNotNone(reference_front_scale)
                self.assertIsNotNone(reference_front_shape)
                self.assertIsNotNone(reference_front_expr)
                self.assertIsNotNone(config.fixed_scale_params)
                self.assertIsNotNone(config.fixed_shape_params)
                np.testing.assert_allclose(config.fixed_scale_params, reference_front_scale, atol=1e-6, rtol=0.0)
                np.testing.assert_allclose(config.fixed_shape_params, reference_front_shape, atol=1e-6, rtol=0.0)

                frame_idx = int(Path(config.out_npy).parent.name)
                out_path = Path(config.out_npy)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_dict = _make_stage1_prediction(seed=frame_idx)
                out_dict["pred_keypoints_3d"] = np.full((70, 3), frame_idx, dtype=np.float32)
                out_dict["pred_vertices"] = np.full((10, 3), frame_idx, dtype=np.float32)
                out_dict["pred_joint_coords"] = np.full((70, 3), frame_idx, dtype=np.float32)
                out_dict["pred_global_rots"] = np.tile(np.eye(3, dtype=np.float32), (70, 1, 1))
                out_dict["opt_is_bad_loss"] = 0
                out_dict["opt_best_loss"] = 1e-6
                out_dict["opt_final_loss"] = 1e-6
                out_dict["opt_best_data_loss"] = 1e-6
                out_dict["opt_final_data_loss"] = 1e-6
                out_dict["opt_best_iter"] = 1
                out_dict["opt_sim_scale"] = 1.0
                out_dict["opt_sim_R"] = np.eye(3, dtype=np.float32)
                out_dict["opt_sim_t"] = np.zeros(3, dtype=np.float32)
                np.save(out_path, out_dict, allow_pickle=True)
                return OptimizationRunResult(
                    out_npy=out_path,
                    debug_dir=out_path.parent / "debug_opt",
                    best_cam="front",
                    loss_history=[1e-6],
                    best_loss=1e-6,
                    final_loss=1e-6,
                    best_data_loss=1e-6,
                    final_data_loss=1e-6,
                    best_iter=1,
                    used_temporal_init=False,
                    is_bad_loss=False,
                    best_pose=np.zeros((133,), dtype=np.float32),
                    sim_scale=1.0,
                    sim_R=np.eye(3, dtype=np.float32),
                    sim_t=np.zeros((3,), dtype=np.float32),
                )

            with mock.patch.object(sequence_runner, "run_demo", side_effect=fail_run_demo), \
                mock.patch.object(sequence_runner, "run_triangulation", side_effect=fail_run_triangulation), \
                mock.patch.object(sequence_runner, "build_optimization_runtime", side_effect=fake_build_runtime), \
                mock.patch.object(sequence_runner, "run_optimization", side_effect=fake_run_optimization):
                config = FullPipelineConfig(
                    image_folder=str(image_root),
                    output_root=str(output_root),
                    cams=cams,
                    caliscope_toml=str(tmp_path / "unused.toml"),
                    checkpoint_path="dummy.ckpt",
                    mhr_path="dummy.pt",
                    hf_repo="fake/repo",
                    device="cpu",
                    min_views=2,
                    skip_inference=True,
                    skip_triangulation=True,
                    fixed_mhr_param_frame_idx=0,
                    fixed_mhr_param_cam="front",
                    overwrite=True,
                    save_sequence_mp4=False,
                )
                result = run_full_pipeline(config)

            self.assertTrue(all(frame.status == "ok" for frame in result.frames))

    def test_recovery_preserves_unrecoverable_bad_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            optimization_root = tmp_path / "optimization"
            out_path = optimization_root / "0" / "opt_out.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_dict = _make_stage1_prediction(seed=0)
            out_dict["opt_is_bad_loss"] = 1
            np.save(out_path, out_dict, allow_pickle=True)

            frame = FramePipelineResult(
                rel_dir="0",
                frame_index=0,
                npy_dir=None,
                available_cams=[],
                used_cams=[],
                triangulated_npz=None,
                optimized_npy=out_path,
                smoothed_npy=None,
                status="bad_loss",
                is_bad_loss=True,
            )

            frame_dicts = recover_missing_and_bad_frames(
                frame_results=[frame],
                optimization_root=optimization_root,
                optimized_name="opt_out.npy",
                max_edge_copy_span=15,
            )

            self.assertTrue(out_path.exists())
            self.assertIsNone(frame_dicts[0])
            self.assertEqual(frame.status, "bad_loss")
            self.assertTrue(frame.is_bad_loss)
            self.assertEqual(frame.optimized_npy, out_path)
            self.assertEqual(int(np.asarray(load_npy_dict(out_path)["opt_is_bad_loss"]).reshape(())), 1)


if __name__ == "__main__":
    unittest.main()
