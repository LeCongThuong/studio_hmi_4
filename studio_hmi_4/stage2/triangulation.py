"""Triangulation and per-point bundle-adjustment logic for stage-2."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .camera import CameraModel
from .io import (
    NP_EXTS,
    find_existing_with_exts,
    invalidate_points_outside_image,
    load_pred_keypoints_2d,
    maybe_denormalize,
)
from .subset import MHRSubset


def huber_rho(r: np.ndarray, delta: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    d = float(delta)
    out = np.empty_like(r)
    mask = r <= d
    out[mask] = 0.5 * r[mask] * r[mask]
    out[~mask] = d * (r[~mask] - 0.5 * d)
    return out


def robust_score(errors: np.ndarray, score_type: str, huber_delta: float) -> float:
    errs = np.asarray(errors, dtype=np.float64)
    if errs.size == 0:
        return float("inf")
    if score_type == "median":
        return float(np.median(errs))
    if score_type == "trimmed":
        if errs.size <= 1:
            return float(errs.mean())
        return float(np.mean(np.sort(errs)[:-1]))
    if score_type == "huber":
        return float(np.sum(huber_rho(errs, delta=huber_delta)))
    raise ValueError(f"Unknown score_type: {score_type}")


class TriangulatorBA:
    """Load observations, triangulate robustly, refine, and save diagnostics."""

    def __init__(
        self,
        cams: List[str],
        cameras: Dict[str, CameraModel],
        subset: MHRSubset,
        npy_dir: str,
        index: int,
        force_normalized: bool,
        force_pixel: bool,
        lm_iters: int,
        lm_lambda: float,
        lm_eps: float,
        score_type: str,
        huber_delta: float,
        inlier_thresh: float,
        robust_lm: bool,
        robust_lm_delta: float,
        reseed_from_inliers: bool,
    ):
        self.cams = cams
        self.cameras = cameras
        self.subset = subset
        self.npy_dir = Path(npy_dir).expanduser().resolve()
        self.index = int(index)
        self.force_normalized = bool(force_normalized)
        self.force_pixel = bool(force_pixel)
        self.lm_iters = int(lm_iters)
        self.lm_lambda = float(lm_lambda)
        self.lm_eps = float(lm_eps)
        self.score_type = score_type
        self.huber_delta = float(huber_delta)
        self.inlier_thresh = float(inlier_thresh)
        self.robust_lm = bool(robust_lm)
        self.robust_lm_delta = float(robust_lm_delta)
        self.reseed_from_inliers = bool(reseed_from_inliers)
        self.M = int(subset.subset_indices.shape[0])

        self.obs_per_cam: Dict[str, np.ndarray] = {}
        self.und_per_cam: Dict[str, np.ndarray] = {}

    def load_observations(self) -> None:
        for cam in self.cams:
            kp_file = find_existing_with_exts(self.npy_dir, cam, NP_EXTS)
            if kp_file is None:
                raise FileNotFoundError(
                    f"Missing keypoints for '{cam}' in {self.npy_dir} (need {cam}.npy or {cam}.npz)"
                )

            k70 = load_pred_keypoints_2d(kp_file, index=self.index)
            ksub = k70[self.subset.subset_indices]

            camera = self.cameras[cam]
            ksub = maybe_denormalize(
                ksub,
                w=camera.w,
                h=camera.h,
                force_normalized=self.force_normalized,
                force_pixel=self.force_pixel,
            )
            ksub = invalidate_points_outside_image(ksub, w=camera.w, h=camera.h)

            self.obs_per_cam[cam] = ksub.astype(np.float64)
            self.und_per_cam[cam] = camera.undistort_to_normalized(ksub)

    def init_by_pair_selection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        V = len(self.cams)
        if V == 3:
            pairs = [(0, 1), (0, 2), (1, 2)]
        else:
            pairs = [(i, j) for i in range(V) for j in range(i + 1, V)]

        points3d_init = np.full((self.M, 3), np.nan, dtype=np.float64)
        best_pair_idx = np.full((self.M, 2), -1, dtype=np.int32)
        inlier_mask = np.zeros((self.M, V), dtype=bool)

        for j in range(self.M):
            obs_uv = [self.obs_per_cam[self.cams[v]][j] for v in range(V)]
            valid_view = np.array([np.isfinite(obs_uv[v]).all() for v in range(V)], dtype=bool)
            if valid_view.sum() < 2:
                continue

            best_X = None
            best_score = float("inf")
            best_pair = (-1, -1)

            for a, b in pairs:
                if not (valid_view[a] and valid_view[b]):
                    continue
                camA = self.cameras[self.cams[a]]
                camB = self.cameras[self.cams[b]]
                xyA = self.und_per_cam[self.cams[a]][j]
                xyB = self.und_per_cam[self.cams[b]][j]

                Xcand = self._triangulate_point_dlt([(xyA, camA.P_norm), (xyB, camB.P_norm)])
                if not np.isfinite(Xcand).all():
                    continue

                errs = []
                for v in range(V):
                    if not valid_view[v]:
                        continue
                    camV = self.cameras[self.cams[v]]
                    uv_hat = camV.project_point_px(Xcand)
                    if not np.isfinite(uv_hat).all():
                        continue
                    residual = float(np.linalg.norm(uv_hat - obs_uv[v]))
                    if np.isfinite(residual) and residual >= 0:
                        errs.append(residual)

                if len(errs) < 2:
                    continue

                score = robust_score(np.array(errs, dtype=np.float64), self.score_type, self.huber_delta)
                if score < best_score:
                    best_score = score
                    best_X = Xcand
                    best_pair = (a, b)

            if best_X is None:
                views = []
                for v in range(V):
                    if not valid_view[v]:
                        continue
                    camV = self.cameras[self.cams[v]]
                    views.append((self.und_per_cam[self.cams[v]][j], camV.P_norm))
                best_X = self._triangulate_point_dlt(views)
                best_pair = (-1, -1)

            best_pair_idx[j] = np.array(best_pair, dtype=np.int32)

            if np.isfinite(best_X).all():
                for v in range(V):
                    if not valid_view[v]:
                        continue
                    camV = self.cameras[self.cams[v]]
                    uv_hat = camV.project_point_px(best_X)
                    if not np.isfinite(uv_hat).all():
                        continue
                    residual = float(np.linalg.norm(uv_hat - obs_uv[v]))
                    if np.isfinite(residual) and residual < self.inlier_thresh:
                        inlier_mask[j, v] = True

                if inlier_mask[j].sum() < 2 and valid_view.sum() >= 2:
                    errs_all = []
                    for v in range(V):
                        if not valid_view[v]:
                            continue
                        camV = self.cameras[self.cams[v]]
                        uv_hat = camV.project_point_px(best_X)
                        residual = (
                            float(np.linalg.norm(uv_hat - obs_uv[v]))
                            if np.isfinite(uv_hat).all()
                            else float("inf")
                        )
                        errs_all.append((residual, v))
                    errs_all.sort(key=lambda item: item[0])
                    keep = [view_idx for _, view_idx in errs_all[:2]]
                    inlier_mask[j, :] = False
                    for view_idx in keep:
                        if np.isfinite(obs_uv[view_idx]).all():
                            inlier_mask[j, view_idx] = True

                if self.reseed_from_inliers and inlier_mask[j].sum() >= 2:
                    dlt_views = []
                    for v in range(V):
                        if not inlier_mask[j, v]:
                            continue
                        xy = self.und_per_cam[self.cams[v]][j]
                        if not np.isfinite(xy).all():
                            continue
                        dlt_views.append((xy, self.cameras[self.cams[v]].P_norm))
                    if len(dlt_views) >= 2:
                        X_reseed = self._triangulate_point_dlt(dlt_views)
                        if np.isfinite(X_reseed).all():
                            best_X = X_reseed

            points3d_init[j] = best_X

        return points3d_init, best_pair_idx, inlier_mask

    def refine_lm_per_point(self, points3d_init: np.ndarray, inlier_mask: np.ndarray) -> np.ndarray:
        V = len(self.cams)
        points3d_ref = np.full((self.M, 3), np.nan, dtype=np.float64)

        for j in range(self.M):
            X0 = points3d_init[j]
            if not np.isfinite(X0).all():
                continue
            points3d_ref[j] = X0

            inliers = inlier_mask[j]
            if inliers.sum() < 2:
                continue

            obs_uvs: List[np.ndarray] = []
            cams_used: List[CameraModel] = []
            dlt_views: List[Tuple[np.ndarray, np.ndarray]] = []

            for v in range(V):
                if not inliers[v]:
                    continue
                cam_name = self.cams[v]
                uv = self.obs_per_cam[cam_name][j]
                if not np.isfinite(uv).all():
                    continue
                obs_uvs.append(uv.astype(np.float64))
                cams_used.append(self.cameras[cam_name])
                und = self.und_per_cam[cam_name][j]
                if np.isfinite(und).all():
                    dlt_views.append((und.astype(np.float64), self.cameras[cam_name].P_norm))

            if len(obs_uvs) < 2:
                continue

            X_seed = X0
            if self.reseed_from_inliers and len(dlt_views) >= 2:
                X_seed_try = self._triangulate_point_dlt(dlt_views)
                if np.isfinite(X_seed_try).all():
                    X_seed = X_seed_try

            Xref = self._lm_refine_point(
                X0=X_seed,
                obs_uvs=obs_uvs,
                cams=cams_used,
                max_iters=self.lm_iters,
                lambda0=self.lm_lambda,
                eps_jac=self.lm_eps,
                robust=self.robust_lm,
                robust_delta=self.robust_lm_delta,
            )
            if np.isfinite(Xref).all():
                points3d_ref[j] = Xref

        return points3d_ref

    @staticmethod
    def _triangulate_point_dlt(xys_norm_and_P: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        if len(xys_norm_and_P) < 2:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        rows = []
        for xy, P in xys_norm_and_P:
            x, y = float(xy[0]), float(xy[1])
            rows.append(x * P[2, :] - P[0, :])
            rows.append(y * P[2, :] - P[1, :])
        A = np.stack(rows, axis=0)

        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1, :]
        if abs(X_h[3]) < 1e-12:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return (X_h[:3] / X_h[3]).astype(np.float64)

    @staticmethod
    def _lm_refine_point(
        X0: np.ndarray,
        obs_uvs: List[np.ndarray],
        cams: List[CameraModel],
        max_iters: int,
        lambda0: float,
        eps_jac: float,
        tol: float = 1e-6,
        robust: bool = False,
        robust_delta: float = 10.0,
    ) -> np.ndarray:
        X = X0.astype(np.float64).copy()
        lam = float(lambda0)
        delta = float(robust_delta)

        def weighted_residual_vec(Xcur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            blocks = []
            weights = []
            for uv_obs, cam in zip(obs_uvs, cams):
                uv_proj = cam.project_point_px(Xcur)
                residual = (uv_proj - uv_obs).astype(np.float64)
                mag = float(np.linalg.norm(residual))
                if robust:
                    weight = 1.0 if mag <= delta else (delta / max(mag, 1e-12))
                else:
                    weight = 1.0
                scale = np.sqrt(weight)
                blocks.append(scale * residual)
                weights.append(weight)
            return np.concatenate(blocks, axis=0), np.array(weights, dtype=np.float64)

        residual, _ = weighted_residual_vec(X)
        if residual.size < 4:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        cost = float(residual @ residual)
        for _ in range(int(max_iters)):
            J = np.zeros((residual.size, 3), dtype=np.float64)
            for k in range(3):
                step = eps_jac * max(1.0, abs(X[k]))
                Xp = X.copy()
                Xp[k] += step
                Xm = X.copy()
                Xm[k] -= step
                rp, _ = weighted_residual_vec(Xp)
                rm, _ = weighted_residual_vec(Xm)
                if rp.size != residual.size or rm.size != residual.size:
                    return X
                J[:, k] = (rp - rm) / (2.0 * step)

            JTJ = J.T @ J
            g = J.T @ residual
            A = JTJ + lam * np.eye(3, dtype=np.float64)

            try:
                dx = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            if np.linalg.norm(dx) < tol:
                break

            X_new = X + dx
            residual_new, _ = weighted_residual_vec(X_new)
            cost_new = float(residual_new @ residual_new)

            if cost_new < cost:
                X = X_new
                residual = residual_new
                if abs(cost - cost_new) / max(1.0, cost) < tol:
                    break
                cost = cost_new
                lam = max(lam * 0.3, 1e-12)
            else:
                lam *= 2.0
                if lam > 1e12:
                    break
        return X

    def per_cam_reprojection(self, points3d: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        proj_per_cam: Dict[str, np.ndarray] = {}
        mean_errs: Dict[str, float] = {}

        for cam in self.cams:
            camera = self.cameras[cam]
            obs = self.obs_per_cam[cam]
            proj = camera.project_points_px(points3d)
            err = np.full((self.M,), np.nan, dtype=np.float64)
            ok = np.isfinite(points3d).all(axis=1) & np.isfinite(obs).all(axis=1) & np.isfinite(proj).all(axis=1)
            if ok.any():
                diff = proj[ok] - obs[ok]
                err[ok] = np.sqrt(np.sum(diff * diff, axis=1))
                mean_errs[cam] = float(np.nanmean(err))
            else:
                mean_errs[cam] = float("nan")
            proj_per_cam[cam] = proj

        return proj_per_cam, mean_errs

    def per_cam_errors(self, points3d: np.ndarray, proj_per_cam: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        err_per_cam: Dict[str, np.ndarray] = {}
        for cam in self.cams:
            obs = self.obs_per_cam[cam]
            proj = proj_per_cam[cam]
            err = np.full((self.M,), np.nan, dtype=np.float64)
            ok = np.isfinite(points3d).all(axis=1) & np.isfinite(obs).all(axis=1) & np.isfinite(proj).all(axis=1)
            if ok.any():
                diff = proj[ok] - obs[ok]
                err[ok] = np.sqrt(np.sum(diff * diff, axis=1))
            err_per_cam[cam] = err
        return err_per_cam

    def save_npz(
        self,
        out_npz: Path,
        points3d_init: np.ndarray,
        points3d_ref: np.ndarray,
        best_pair_idx: np.ndarray,
        inlier_mask: np.ndarray,
        mean_init: Dict[str, float],
        mean_ref: Dict[str, float],
        proj_init: Dict[str, np.ndarray],
        proj_ref: Dict[str, np.ndarray],
        err_init: Dict[str, np.ndarray],
        err_ref: Dict[str, np.ndarray],
        toml_sections: List[str],
    ) -> None:
        save_dict = {
            "subset_names": self.subset.subset_names,
            "subset_indices": self.subset.subset_indices,
            "cams": np.array(self.cams, dtype=object),
            "toml_sections": np.array(toml_sections, dtype=object),
            "points3d_init": points3d_init,
            "points3d_refined": points3d_ref,
            "best_pair_idx": best_pair_idx,
            "inlier_mask": inlier_mask.astype(np.uint8),
            "score_type": np.array(self.score_type, dtype=object),
            "huber_delta": np.array(self.huber_delta, dtype=np.float64),
            "inlier_thresh": np.array(self.inlier_thresh, dtype=np.float64),
            "robust_lm": np.array(int(self.robust_lm), dtype=np.int32),
            "robust_lm_delta": np.array(self.robust_lm_delta, dtype=np.float64),
        }

        for cam in self.cams:
            save_dict[f"{cam}_kpts2d_obs"] = self.obs_per_cam[cam]

        for cam in self.cams:
            save_dict[f"{cam}_kpts2d_proj_init"] = proj_init[cam]
            save_dict[f"{cam}_err_px_init"] = err_init[cam]
            save_dict[f"{cam}_mean_err_px_init"] = np.array(mean_init[cam], dtype=np.float64)
            save_dict[f"{cam}_kpts2d_proj_refined"] = proj_ref[cam]
            save_dict[f"{cam}_err_px_refined"] = err_ref[cam]
            save_dict[f"{cam}_mean_err_px_refined"] = np.array(mean_ref[cam], dtype=np.float64)

        np.savez_compressed(out_npz, **save_dict)
