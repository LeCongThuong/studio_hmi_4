#!/usr/bin/env python3
"""
triangulate_mhr3d_gt.py

Triangulate + per-point Bundle Adjustment (Levenberg–Marquardt) for an MHR-70 subset
(hands + shoulders + elbows + wrists) from multiple views.

Input layout (ONE dir each):
  npy_dir/
    front.npy (or .npz)
    left.npy
    right.npy
  (optional) img_dir/ for saving overlay images (NOT interactive)
    front.jpg / .png
    left.jpg
    right.jpg

Each npy/npz must contain key:
  data["pred_keypoints_2d"]   shape (70,2) or (N,70,2)

Caliscope TOML has camera blocks (e.g., cam_1, cam_2, cam_3) with:
  matrix (K), distortions (D), rotation (rvec Rodrigues), translation (tvec), size=[w,h]

Algorithm:
  1) Select subset indices by names from mhr_70.py pose_info
  2) For each keypoint:
      Step A: Triangulate from EACH PAIR of cameras => candidates
      Step B: Score each candidate by reprojection errors across available views (robust)
              pick best X*
      Step C: Compute per-view errors for X*; keep inlier views (err < tau)
              run BA (LM) using only inlier views (or robust LM if enabled)
  3) Reproject to each camera and compute per-point error
  4) Save everything to out_npz

Debug:
  --debug -> show ONLY interactive 3D scatter of refined points
  Optional: --debug_dir (+ --img_dir) -> save 2D overlays to disk (not interactive)

Example:
  python triangulate_mhr3d_gt.py \
    --mhr_py /path/to/mhr_70.py \
    --caliscope_toml /path/to/config.toml \
    --cams front left right \
    --toml_sections cam_1 cam_2 cam_3 \
    --npy_dir /data/npy \
    --out_npz /data/out/triangulated_ba.npz \
    --debug

If reprojection is insane, try:
  --invert_extrinsics
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import cv2
import numpy as np

# Python 3.11+ has tomllib; fallback to tomli if needed
try:
    import tomllib  # type: ignore
except Exception:
    import tomli as tomllib  # type: ignore

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
NP_EXTS = [".npy", ".npz"]


@dataclass
class TriangulationConfig:
    """Configuration for stage-2 triangulation + per-point bundle adjustment."""

    mhr_py: str
    caliscope_toml: str
    cams: List[str]
    npy_dir: str
    out_npz: str
    toml_sections: Optional[List[str]] = None
    index: int = 0
    normalized: bool = False
    pixel: bool = False
    invert_extrinsics: bool = False
    lm_iters: int = 25
    lm_lambda: float = 1e-3
    lm_eps: float = 1e-4
    debug: bool = False
    debug_dir: Optional[str] = None
    img_dir: Optional[str] = None
    score_type: str = "median"
    huber_delta: float = 10.0
    inlier_thresh: float = 30.0
    robust_lm: bool = False
    robust_lm_delta: float = 10.0
    reseed_from_inliers: bool = True


@dataclass
class TriangulationRunResult:
    """Outputs from stage-2 triangulation."""

    out_npz: Path
    points3d_init: np.ndarray
    points3d_refined: np.ndarray
    mean_err_init: Dict[str, float]
    mean_err_refined: Dict[str, float]
    debug_dir: Optional[Path]


# -----------------------------
# Utilities
# -----------------------------

def import_py_module(py_path: str):
    """Import a .py file as a module."""
    py_path = str(Path(py_path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("mhr_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_existing_with_exts(dir_path: Path, stem: str, exts: List[str]) -> Optional[Path]:
    """Find stem+ext in a directory, supporting multiple possible extensions."""
    exts_lower = [e.lower() for e in exts]
    for ext in exts:
        p = dir_path / f"{stem}{ext}"
        if p.exists():
            return p
    for p in dir_path.glob(stem + ".*"):
        if p.suffix.lower() in exts_lower:
            return p
    return None


def load_pred_keypoints_2d(file_path: Path, index: int) -> np.ndarray:
    """
    Load 2D keypoints from:
      - .npy saved as dict via allow_pickle=True and .item()
      - .npz with key 'pred_keypoints_2d' OR dict stored as arr_0 object

    Returns: (70,2) float64
    """
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        obj = np.load(str(file_path), allow_pickle=True)
        data = obj.item() if hasattr(obj, "item") else obj
    elif suffix == ".npz":
        npz = np.load(str(file_path), allow_pickle=True)
        if "pred_keypoints_2d" in npz:
            data = {"pred_keypoints_2d": npz["pred_keypoints_2d"]}
        elif "arr_0" in npz:
            arr0 = npz["arr_0"]
            data = arr0.item() if hasattr(arr0, "item") else {"pred_keypoints_2d": arr0}
        else:
            raise KeyError(f"{file_path} has keys {list(npz.keys())}, expected 'pred_keypoints_2d'")
    else:
        raise ValueError(f"Unsupported file: {file_path}")

    if "pred_keypoints_2d" not in data:
        raise KeyError(f"{file_path} missing 'pred_keypoints_2d'. Available keys: {list(data.keys())}")

    arr = np.asarray(data["pred_keypoints_2d"])
    if arr.ndim == 2:
        if arr.shape != (70, 2):
            raise ValueError(f"{file_path}: expected (70,2), got {arr.shape}")
        return arr.astype(np.float64)

    if arr.ndim == 3:
        if arr.shape[1:] != (70, 2):
            raise ValueError(f"{file_path}: expected (N,70,2), got {arr.shape}")
        index = int(np.clip(index, 0, arr.shape[0] - 1))
        return arr[index].astype(np.float64)

    raise ValueError(f"{file_path}: expected ndim 2 or 3, got shape {arr.shape}")


def maybe_denormalize(kpts_xy: np.ndarray, w: int, h: int, force_normalized: bool, force_pixel: bool) -> np.ndarray:
    """
    If keypoints look normalized in [0,1], convert to pixel coords using (w,h).
    """
    if force_pixel:
        return kpts_xy.astype(np.float64)

    k = kpts_xy.astype(np.float64).copy()
    finite = np.isfinite(k).all(axis=1)
    if not finite.any():
        return k

    kf = k[finite]
    looks_norm = (
        (kf[:, 0].min() >= -0.5 and kf[:, 0].max() <= 1.5) and
        (kf[:, 1].min() >= -0.5 and kf[:, 1].max() <= 1.5)
    )
    if force_normalized or looks_norm:
        k[:, 0] *= float(w)
        k[:, 1] *= float(h)
    return k


def read_image(img_dir: Path, cam: str, fallback_size_wh: Tuple[int, int]) -> np.ndarray:
    """Read cam image if exists, else return black canvas."""
    p = find_existing_with_exts(img_dir, cam, IMG_EXTS)
    if p is None:
        w, h = fallback_size_wh
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        w, h = fallback_size_wh
        return np.zeros((h, w, 3), dtype=np.uint8)
    return img


def show_3d_scatter_interactive(points3d: np.ndarray, title: str) -> None:
    """
    Interactive 3D scatter. If your env forces Agg, we try switching to a GUI backend.
    """
    try:
        import matplotlib
        # If backend is non-interactive, try a couple of common GUI backends.
        if matplotlib.get_backend().lower() in ("agg", "cairo", "pdf", "svg", "ps"):
            for b in ("TkAgg", "Qt5Agg", "QtAgg"):
                try:
                    matplotlib.use(b, force=True)
                    break
                except Exception:
                    pass
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib interactive show failed to init: {e}")
        return

    X = points3d
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if X.shape[0] == 0:
        print("[DEBUG] No finite 3D points to show.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=14)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# -----------------------------
# MHR subset selection
# -----------------------------

@dataclass
class MHRSubset:
    """
    Subset metadata:
      - subset_names: list of names for chosen subset (order matters)
      - subset_indices: indices into full MHR70 array
      - edges: skeleton edges within subset (indices in subset space)
    """
    subset_names: np.ndarray          # (M,) dtype object
    subset_indices: np.ndarray        # (M,) int32
    edges: List[Tuple[int, int]]      # subset-space edges


class MHRSubsetSelector:
    """
    Build name->index mapping and subset indices for:
      right_hand + left_hand + (shoulders, elbows, wrists)
    """
    def __init__(self, mhr_py: str):
        mod = import_py_module(mhr_py)
        pose_info = getattr(mod, "pose_info", None)
        if pose_info is None:
            raise AttributeError("mhr_70.py must define `pose_info` dict")
        self.pose_info: Dict = pose_info
        self.name_to_idx: Dict[str, int] = self._build_name_to_idx(pose_info)

    @staticmethod
    def _build_name_to_idx(pose_info: Dict) -> Dict[str, int]:
        """
        Fix naming mismatch:
          - keypoint_info[*]['name'] includes right_thumb4 / right_forefinger4 / ...
          - original_keypoint_info includes right_thumb_tip / right_index_tip / ...
        Use keypoint_info first, then add aliases from original_keypoint_info.
        """
        name_to_idx: Dict[str, int] = {}

        kinfo = pose_info.get("keypoint_info", {})
        if isinstance(kinfo, dict) and len(kinfo) > 0:
            for idx_key, v in kinfo.items():
                idx = int(idx_key)
                if isinstance(v, dict) and "name" in v:
                    name_to_idx[str(v["name"])] = idx

        orig = pose_info.get("original_keypoint_info", {})
        if isinstance(orig, dict):
            for idx, nm in orig.items():
                name_to_idx[str(nm)] = int(idx)

        return name_to_idx

    def build_subset(self) -> MHRSubset:
        left_hand = self.pose_info.get("left_hand_keypoint_names", [])
        right_hand = self.pose_info.get("right_hand_keypoint_names", [])
        must = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]

        subset_names = list(right_hand) + list(left_hand) + must

        idxs: List[int] = []
        for n in subset_names:
            if n not in self.name_to_idx:
                sample = sorted(list(self.name_to_idx.keys()))[:40]
                raise KeyError(f"Keypoint name not found: {n}\nExample available names: {sample} ...")
            idxs.append(int(self.name_to_idx[n]))

        subset_names_arr = np.array(subset_names, dtype=object)
        subset_indices = np.array(idxs, dtype=np.int32)

        edges = self._build_subset_edges(subset_indices)
        return MHRSubset(subset_names=subset_names_arr, subset_indices=subset_indices, edges=edges)

    def _build_subset_edges(self, subset_indices: np.ndarray) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        sk = self.pose_info.get("skeleton_info", None)
        if not isinstance(sk, dict):
            return edges

        subset_set = set(subset_indices.tolist())
        full_to_sub = {int(full): i for i, full in enumerate(subset_indices.tolist())}

        for _, e in sk.items():
            a_name, b_name = e["link"]
            if (a_name in self.name_to_idx) and (b_name in self.name_to_idx):
                ia = int(self.name_to_idx[a_name])
                ib = int(self.name_to_idx[b_name])
                if ia in subset_set and ib in subset_set:
                    edges.append((full_to_sub[ia], full_to_sub[ib]))
        return edges


# -----------------------------
# Camera rig
# -----------------------------

@dataclass
class CameraModel:
    """
    Calibrated pinhole camera with distortion.

    Stored:
      - K (3x3), D (nx1)
      - rvec,tvec (Rodrigues) s.t. X_cam = R*X_world + t
      - P_norm (3x4) = [R|t] for normalized DLT triangulation using undistortPoints() coords
      - size w,h
    """
    name: str
    section: str
    K: np.ndarray
    D: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    P_norm: np.ndarray
    w: int
    h: int

    @staticmethod
    def from_caliscope_block(name: str, section: str, block: Dict, invert_extrinsics: bool) -> "CameraModel":
        K = np.array(block["matrix"], dtype=np.float64).reshape(3, 3)
        D = np.array(block["distortions"], dtype=np.float64).reshape(-1, 1)
        rvec = np.array(block["rotation"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(block["translation"], dtype=np.float64).reshape(3, 1)

        if invert_extrinsics:
            rvec, tvec = CameraModel._invert_extrinsics(rvec, tvec)

        size = block.get("size", None)
        if size is None or len(size) != 2:
            raise KeyError(f"Camera '{section}' missing size=[w,h]")
        w_img, h_img = int(size[0]), int(size[1])

        R, _ = cv2.Rodrigues(rvec)
        P_norm = np.hstack([R, tvec.reshape(3, 1)])  # 3x4

        return CameraModel(
            name=name, section=section, K=K, D=D,
            rvec=rvec, tvec=tvec, P_norm=P_norm,
            w=w_img, h=h_img
        )

    @staticmethod
    def _invert_extrinsics(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        t_inv = -R_inv @ tvec
        rvec_inv, _ = cv2.Rodrigues(R_inv)
        return rvec_inv.reshape(3, 1), t_inv.reshape(3, 1)

    def undistort_to_normalized(self, pts_px: np.ndarray) -> np.ndarray:
        pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
        und = cv2.undistortPoints(pts, self.K, self.D)
        return und.reshape(-1, 2)

    def project_points_px(self, Xs_world: np.ndarray) -> np.ndarray:
        Xw = Xs_world.reshape(-1, 1, 3).astype(np.float64)
        uv, _ = cv2.projectPoints(Xw, self.rvec, self.tvec, self.K, self.D)
        return uv.reshape(-1, 2).astype(np.float64)

    def project_point_px(self, X_world: np.ndarray) -> np.ndarray:
        Xw = X_world.reshape(1, 1, 3).astype(np.float64)
        uv, _ = cv2.projectPoints(Xw, self.rvec, self.tvec, self.K, self.D)
        return uv.reshape(2).astype(np.float64)


class CaliscopeRig:
    """Parse a Caliscope TOML and produce CameraModel objects."""
    def __init__(self, caliscope_toml: str):
        toml_path = Path(caliscope_toml).expanduser().resolve()
        self._raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))

    def build_cameras(
        self,
        cams: List[str],
        toml_sections: List[str],
        invert_extrinsics: bool,
    ) -> Dict[str, CameraModel]:
        if len(cams) != len(toml_sections):
            raise ValueError("--toml_sections must match length of --cams (or omit).")

        out: Dict[str, CameraModel] = {}
        for cam_name, section in zip(cams, toml_sections):
            if section not in self._raw:
                raise KeyError(f"TOML section '{section}' not found. Available: {list(self._raw.keys())}")
            block = self._raw[section]
            out[cam_name] = CameraModel.from_caliscope_block(
                name=cam_name, section=section, block=block, invert_extrinsics=invert_extrinsics
            )
        return out


# -----------------------------
# Robust scoring helpers
# -----------------------------

def huber_rho(r: np.ndarray, delta: float) -> np.ndarray:
    """
    Huber penalty on scalar residual magnitude r (>=0):
      0.5 r^2                     if r <= delta
      delta*(r - 0.5*delta)       if r >  delta
    """
    r = np.asarray(r, dtype=np.float64)
    d = float(delta)
    out = np.empty_like(r)
    m = r <= d
    out[m] = 0.5 * r[m] * r[m]
    out[~m] = d * (r[~m] - 0.5 * d)
    return out


def robust_score(errors: np.ndarray, score_type: str, huber_delta: float) -> float:
    """
    errors: (K,) non-negative, finite
    score_type:
      - 'median'  : median(errors)
      - 'trimmed' : drop the single largest error, average the rest
      - 'huber'   : sum huber_rho(errors, delta)
    """
    e = np.asarray(errors, dtype=np.float64)
    if e.size == 0:
        return float("inf")
    if score_type == "median":
        return float(np.median(e))
    if score_type == "trimmed":
        if e.size <= 1:
            return float(e.mean())
        # drop top-1
        e_sorted = np.sort(e)
        return float(np.mean(e_sorted[:-1]))
    if score_type == "huber":
        return float(np.sum(huber_rho(e, delta=huber_delta)))
    raise ValueError(f"Unknown score_type: {score_type}")


# -----------------------------
# Triangulation + BA
# -----------------------------

class TriangulatorBA:
    """
    End-to-end:
      - load per-camera 2D
      - Step A/B: pair-candidate init per keypoint
      - Step C: inlier-view gating
      - LM refinement per keypoint (optionally robust)
      - reprojection + errors
      - save outputs
    """
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
        # robust init/gating
        score_type: str,
        huber_delta: float,
        inlier_thresh: float,
        # robust LM option
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

        # filled later
        self.obs_per_cam: Dict[str, np.ndarray] = {}
        self.und_per_cam: Dict[str, np.ndarray] = {}

    def load_observations(self) -> None:
        """Load per-camera (70,2), select subset, optionally denormalize, undistort to normalized."""
        for cam in self.cams:
            kp_file = find_existing_with_exts(self.npy_dir, cam, NP_EXTS)
            if kp_file is None:
                raise FileNotFoundError(
                    f"Missing keypoints for '{cam}' in {self.npy_dir} (need {cam}.npy or {cam}.npz)"
                )

            k70 = load_pred_keypoints_2d(kp_file, index=self.index)  # (70,2)
            ksub = k70[self.subset.subset_indices]  # (M,2)

            cm = self.cameras[cam]
            ksub = maybe_denormalize(
                ksub, w=cm.w, h=cm.h,
                force_normalized=self.force_normalized,
                force_pixel=self.force_pixel
            )

            self.obs_per_cam[cam] = ksub.astype(np.float64)
            self.und_per_cam[cam] = cm.undistort_to_normalized(ksub)

    # ---------- Step A/B/C init ----------
    def init_by_pair_selection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each keypoint j:
          Step A: triangulate from each camera pair -> candidates
          Step B: score each candidate by reprojection errors across available views (robust)
          Pick best X*.
          Step C: inlier view mask based on err(X*) < inlier_thresh.

        Returns:
          points3d_init: (M,3)
          best_pair_idx: (M,2) indices into self.cams, -1 if invalid
          inlier_mask:   (M,V) bool
        """
        V = len(self.cams)
        pairs = [(0, 1), (0, 2), (1, 2)] if V == 3 else [(i, j) for i in range(V) for j in range(i + 1, V)]

        points3d_init = np.full((self.M, 3), np.nan, dtype=np.float64)
        best_pair_idx = np.full((self.M, 2), -1, dtype=np.int32)
        inlier_mask = np.zeros((self.M, V), dtype=bool)

        for j in range(self.M):
            # collect which views are available
            obs_uv = [self.obs_per_cam[self.cams[v]][j] for v in range(V)]
            valid_view = np.array([np.isfinite(obs_uv[v]).all() for v in range(V)], dtype=bool)

            if valid_view.sum() < 2:
                continue

            best_X = None
            best_score = float("inf")
            best_pair = (-1, -1)

            # Step A: candidates from each pair
            for (a, b) in pairs:
                if not (valid_view[a] and valid_view[b]):
                    continue

                camA = self.cameras[self.cams[a]]
                camB = self.cameras[self.cams[b]]
                xyA = self.und_per_cam[self.cams[a]][j]
                xyB = self.und_per_cam[self.cams[b]][j]

                Xcand = self._triangulate_point_dlt([(xyA, camA.P_norm), (xyB, camB.P_norm)])
                if not np.isfinite(Xcand).all():
                    continue

                # Step B: robust scoring using reprojection errors across ALL valid views
                errs = []
                for v in range(V):
                    if not valid_view[v]:
                        continue
                    camV = self.cameras[self.cams[v]]
                    uv_hat = camV.project_point_px(Xcand)
                    if not np.isfinite(uv_hat).all():
                        continue
                    r = float(np.linalg.norm(uv_hat - obs_uv[v]))
                    if np.isfinite(r) and r >= 0:
                        errs.append(r)

                if len(errs) < 2:
                    continue

                s = robust_score(np.array(errs, dtype=np.float64), self.score_type, self.huber_delta)
                if s < best_score:
                    best_score = s
                    best_X = Xcand
                    best_pair = (a, b)

            if best_X is None:
                # fallback: naive all-view DLT (your original init)
                views = []
                for v in range(V):
                    if not valid_view[v]:
                        continue
                    camV = self.cameras[self.cams[v]]
                    xy = self.und_per_cam[self.cams[v]][j]
                    views.append((xy, camV.P_norm))
                best_X = self._triangulate_point_dlt(views)
                best_pair = (-1, -1)

            best_pair_idx[j] = np.array(best_pair, dtype=np.int32)

            # Step C: inlier selection based on X*
            if np.isfinite(best_X).all():
                for v in range(V):
                    if not valid_view[v]:
                        continue
                    camV = self.cameras[self.cams[v]]
                    uv_hat = camV.project_point_px(best_X)
                    if not np.isfinite(uv_hat).all():
                        continue
                    r = float(np.linalg.norm(uv_hat - obs_uv[v]))
                    if np.isfinite(r) and (r < self.inlier_thresh):
                        inlier_mask[j, v] = True

                # guarantee at least 2 inliers if possible: if too strict, keep the best 2 views
                if inlier_mask[j].sum() < 2 and valid_view.sum() >= 2:
                    # pick two smallest reprojection errors among valid views
                    errs_all = []
                    for v in range(V):
                        if not valid_view[v]:
                            continue
                        camV = self.cameras[self.cams[v]]
                        uv_hat = camV.project_point_px(best_X)
                        r = float(np.linalg.norm(uv_hat - obs_uv[v])) if np.isfinite(uv_hat).all() else float("inf")
                        errs_all.append((r, v))
                    errs_all.sort(key=lambda x: x[0])
                    keep = [vv for _, vv in errs_all[:2]]
                    inlier_mask[j, :] = False
                    for vv in keep:
                        if np.isfinite(obs_uv[vv]).all():
                            inlier_mask[j, vv] = True

                # optional reseed: DLT from all selected inlier views
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

    # ---------- refinement ----------
    def refine_lm_per_point(self, points3d_init: np.ndarray, inlier_mask: np.ndarray) -> np.ndarray:
        """
        Run per-point LM using only views marked as inliers for that point.
        If robust_lm=True, apply Huber weights inside LM (approx IRLS).
        """
        V = len(self.cams)
        points3d_ref = np.full((self.M, 3), np.nan, dtype=np.float64)

        for j in range(self.M):
            X0 = points3d_init[j]
            if not np.isfinite(X0).all():
                continue
            # preserve initialized estimate when refinement is not possible
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

        A_rows = []
        for xy, P in xys_norm_and_P:
            x, y = float(xy[0]), float(xy[1])
            A_rows.append(x * P[2, :] - P[0, :])
            A_rows.append(y * P[2, :] - P[1, :])
        A = np.stack(A_rows, axis=0)

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
        """
        Minimize reprojection errors with LM on 3 parameters.

        If robust=True:
          we apply per-view Huber weights on residual blocks (2D),
          implemented by scaling residuals by sqrt(w) each iteration.
          This is a lightweight IRLS-ish robust LM (good enough for hands).
        """
        X = X0.astype(np.float64).copy()
        lam = float(lambda0)
        d = float(robust_delta)

        def residual_vec(Xcur: np.ndarray) -> np.ndarray:
            rs = []
            for uv_obs, cam in zip(obs_uvs, cams):
                uv_proj = cam.project_point_px(Xcur)
                rs.append(uv_proj - uv_obs)
            return np.concatenate(rs, axis=0)  # (2V,)

        def weighted_residual_vec(Xcur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Returns:
              r_w : weighted residual vector (2V,)
              w2  : per-block weights (V,) (for debugging)
            """
            blocks = []
            w_blocks = []
            for uv_obs, cam in zip(obs_uvs, cams):
                uv_proj = cam.project_point_px(Xcur)
                e = (uv_proj - uv_obs).astype(np.float64)
                r = float(np.linalg.norm(e))
                if robust:
                    # Huber weight on magnitude
                    w = 1.0 if r <= d else (d / max(r, 1e-12))
                else:
                    w = 1.0
                s = np.sqrt(w)
                blocks.append(s * e)
                w_blocks.append(w)
            return np.concatenate(blocks, axis=0), np.array(w_blocks, dtype=np.float64)

        # initialize
        r, _ = weighted_residual_vec(X)
        if r.size < 4:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        cost = float(r @ r)

        for _ in range(int(max_iters)):
            # numerical Jacobian on weighted residual
            J = np.zeros((r.size, 3), dtype=np.float64)
            for k in range(3):
                step = eps_jac * max(1.0, abs(X[k]))
                Xp = X.copy(); Xp[k] += step
                Xm = X.copy(); Xm[k] -= step

                rp, _ = weighted_residual_vec(Xp)
                rm, _ = weighted_residual_vec(Xm)

                if rp.size != r.size or rm.size != r.size:
                    return X
                J[:, k] = (rp - rm) / (2.0 * step)

            JTJ = J.T @ J
            g = J.T @ r
            A = JTJ + lam * np.eye(3, dtype=np.float64)

            try:
                dx = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            if np.linalg.norm(dx) < tol:
                break

            X_new = X + dx
            r_new, _ = weighted_residual_vec(X_new)
            cost_new = float(r_new @ r_new)

            if cost_new < cost:
                X = X_new
                r = r_new
                if abs(cost - cost_new) / max(1.0, cost) < tol:
                    cost = cost_new
                    break
                cost = cost_new
                lam = max(lam * 0.3, 1e-12)
            else:
                lam *= 2.0
                if lam > 1e12:
                    break

        return X

    # ---------- reprojection/errors ----------
    def per_cam_reprojection(self, points3d: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Returns:
          proj_per_cam: {cam: (M,2)}
          mean_errs:    {cam: float}
        """
        proj_per_cam: Dict[str, np.ndarray] = {}
        mean_errs: Dict[str, float] = {}

        for cam in self.cams:
            cm = self.cameras[cam]
            obs = self.obs_per_cam[cam]
            proj = cm.project_points_px(points3d)

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

    # ---------- saving ----------
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
        """
        Save outputs with your existing key style + extra debug keys.
        """
        save_dict = {
            "subset_names": self.subset.subset_names,
            "subset_indices": self.subset.subset_indices,
            "cams": np.array(self.cams, dtype=object),
            "toml_sections": np.array(toml_sections, dtype=object),

            "points3d_init": points3d_init,
            "points3d_refined": points3d_ref,

            # extra helpful metadata (won't break anything)
            "best_pair_idx": best_pair_idx,          # (M,2) camera indices into cams list, -1 fallback
            "inlier_mask": inlier_mask.astype(np.uint8),  # (M,V) 0/1
            "score_type": np.array(self.score_type, dtype=object),
            "huber_delta": np.array(self.huber_delta, dtype=np.float64),
            "inlier_thresh": np.array(self.inlier_thresh, dtype=np.float64),
            "robust_lm": np.array(int(self.robust_lm), dtype=np.int32),
            "robust_lm_delta": np.array(self.robust_lm_delta, dtype=np.float64),
        }

        # per-cam obs
        for cam in self.cams:
            save_dict[f"{cam}_kpts2d_obs"] = self.obs_per_cam[cam]

        # per-cam projections/errors + means
        for cam in self.cams:
            save_dict[f"{cam}_kpts2d_proj_init"] = proj_init[cam]
            save_dict[f"{cam}_err_px_init"] = err_init[cam]
            save_dict[f"{cam}_mean_err_px_init"] = np.array(mean_init[cam], dtype=np.float64)

            save_dict[f"{cam}_kpts2d_proj_refined"] = proj_ref[cam]
            save_dict[f"{cam}_err_px_refined"] = err_ref[cam]
            save_dict[f"{cam}_mean_err_px_refined"] = np.array(mean_ref[cam], dtype=np.float64)

        np.savez_compressed(out_npz, **save_dict)


# -----------------------------
# Overlay drawing (unchanged semantics)
# -----------------------------

def draw_overlay(img_bgr: np.ndarray, obs: np.ndarray, proj: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Red   = observed/input 2D
    Green = projected 2D (from 3D)
    White = error vector (obs -> proj)
    """
    out = img_bgr.copy()

    def valid_2d(pt: np.ndarray) -> bool:
        return bool(np.isfinite(pt).all())

    for a, b in edges:
        if valid_2d(obs[a]) and valid_2d(obs[b]):
            cv2.line(out, tuple(obs[a].astype(int)), tuple(obs[b].astype(int)), (0, 0, 180), 1)
        if valid_2d(proj[a]) and valid_2d(proj[b]):
            cv2.line(out, tuple(proj[a].astype(int)), tuple(proj[b].astype(int)), (0, 180, 0), 1)

    for i in range(obs.shape[0]):
        if valid_2d(obs[i]):
            cv2.circle(out, tuple(obs[i].astype(int)), 3, (0, 0, 255), -1)
        if valid_2d(proj[i]):
            cv2.circle(out, tuple(proj[i].astype(int)), 3, (0, 255, 0), -1)
        if valid_2d(obs[i]) and valid_2d(proj[i]):
            cv2.line(out, tuple(obs[i].astype(int)), tuple(proj[i].astype(int)), (255, 255, 255), 1)

    return out


# -----------------------------
# CLI (keep existing args; add optional robust args)
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone triangulation runs."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--mhr_py", required=True, help="Path to mhr_70.py (defines pose_info)")
    ap.add_argument("--caliscope_toml", required=True, help="Path to Caliscope config.toml")
    ap.add_argument("--cams", nargs="+", required=True, help="Camera names (= filename stems): front left right")
    ap.add_argument("--toml_sections", nargs="*", default=None,
                    help="Optional TOML section names aligned with --cams: e.g. cam_1 cam_2 cam_3")
    ap.add_argument("--npy_dir", required=True, help="Directory with per-cam npy/npz (front.npy, ...)")
    ap.add_argument("--out_npz", required=True, help="Output .npz path")
    ap.add_argument("--index", type=int, default=0, help="Frame index if pred_keypoints_2d is (N,70,2)")
    ap.add_argument("--normalized", action="store_true", help="Force treat input 2D as normalized [0,1]")
    ap.add_argument("--pixel", action="store_true", help="Force treat input 2D as pixel coords")
    ap.add_argument("--invert_extrinsics", action="store_true",
                    help="Try if reprojection is wrong (treat stored extrinsics as cam->world)")
    # LM params
    ap.add_argument("--lm_iters", type=int, default=25, help="LM iterations per point")
    ap.add_argument("--lm_lambda", type=float, default=1e-3, help="Initial LM damping")
    ap.add_argument("--lm_eps", type=float, default=1e-4, help="Finite-difference step scale")
    # Debug
    ap.add_argument("--debug", action="store_true", help="Show interactive 3D scatter (refined points)")
    ap.add_argument("--debug_dir", default=None, help="If set, save overlay images and a 3D scatter png (no interactive 2D)")
    ap.add_argument("--img_dir", default=None, help="Optional: image dir for saving overlays (front.jpg, ...)")

    # --- NEW optional robust args (do not break old calls) ---
    ap.add_argument("--score_type", type=str, default="median",
                    choices=["median", "trimmed", "huber"],
                    help="Robust candidate scoring for pair init: median | trimmed | huber")
    ap.add_argument("--huber_delta", type=float, default=10.0,
                    help="Delta for huber score (pixels), only used if --score_type huber")
    ap.add_argument("--inlier_thresh", type=float, default=30.0,
                    help="Inlier threshold tau (pixels) for selecting views before BA")
    ap.add_argument("--robust_lm", action="store_true",
                    help="If set, use Huber-weighted residuals inside LM (robust BA)")
    ap.add_argument("--robust_lm_delta", type=float, default=10.0,
                    help="Delta for robust LM Huber weighting (pixels)")
    ap.add_argument(
        "--no_reseed_from_inliers",
        action="store_true",
        help="Disable DLT reseeding from selected inlier views before LM refinement.",
    )
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args for stage-2 triangulation."""

    parser = build_arg_parser()
    return parser.parse_args(argv)


def resolve_toml_sections(
    cams: List[str],
    toml_sections_arg: Optional[List[str]],
) -> List[str]:
    """Resolve TOML camera section names aligned with logical `cams` order."""

    if toml_sections_arg is None or len(toml_sections_arg) == 0:
        return cams[:]

    if len(toml_sections_arg) != len(cams):
        raise ValueError("--toml_sections must match length of --cams (or omit).")
    return toml_sections_arg


def namespace_to_config(args: argparse.Namespace) -> TriangulationConfig:
    """Convert parsed CLI args to `TriangulationConfig`."""

    return TriangulationConfig(
        mhr_py=args.mhr_py,
        caliscope_toml=args.caliscope_toml,
        cams=list(args.cams),
        toml_sections=args.toml_sections,
        npy_dir=args.npy_dir,
        out_npz=args.out_npz,
        index=args.index,
        normalized=args.normalized,
        pixel=args.pixel,
        invert_extrinsics=args.invert_extrinsics,
        lm_iters=args.lm_iters,
        lm_lambda=args.lm_lambda,
        lm_eps=args.lm_eps,
        debug=args.debug,
        debug_dir=args.debug_dir,
        img_dir=args.img_dir,
        score_type=args.score_type,
        huber_delta=args.huber_delta,
        inlier_thresh=args.inlier_thresh,
        robust_lm=args.robust_lm,
        robust_lm_delta=args.robust_lm_delta,
        reseed_from_inliers=not args.no_reseed_from_inliers,
    )


def run_triangulation(config: TriangulationConfig) -> TriangulationRunResult:
    """Execute stage-2 triangulation and persist a `.npz` bundle.

    Core idea:
    1. Load per-camera 2D keypoints and select the target MHR subset.
    2. Generate robust pairwise triangulation hypotheses.
    3. Gate inlier views and run per-point LM refinement.
    4. Save refined 3D points, reprojections, and diagnostics to disk.
    """

    cams = list(config.cams)
    if len(cams) < 2:
        raise ValueError("Need at least 2 cameras for triangulation/BA.")
    if len(cams) == 3:
        # exactly what you said you have; pairs are hard-coded efficiently in init
        pass

    toml_sections = resolve_toml_sections(cams, config.toml_sections)

    out_npz = Path(config.out_npz).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if config.debug_dir is not None:
        debug_dir = Path(config.debug_dir).expanduser().resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(config.img_dir).expanduser().resolve() if config.img_dir is not None else None

    # 1) subset
    selector = MHRSubsetSelector(config.mhr_py)
    subset = selector.build_subset()

    # 2) cameras
    rig = CaliscopeRig(config.caliscope_toml)
    cameras = rig.build_cameras(
        cams=cams,
        toml_sections=toml_sections,
        invert_extrinsics=config.invert_extrinsics,
    )

    # 3) pipeline
    pipe = TriangulatorBA(
        cams=cams,
        cameras=cameras,
        subset=subset,
        npy_dir=config.npy_dir,
        index=config.index,
        force_normalized=config.normalized,
        force_pixel=config.pixel,
        lm_iters=config.lm_iters,
        lm_lambda=config.lm_lambda,
        lm_eps=config.lm_eps,
        score_type=config.score_type,
        huber_delta=config.huber_delta,
        inlier_thresh=config.inlier_thresh,
        robust_lm=config.robust_lm,
        robust_lm_delta=config.robust_lm_delta,
        reseed_from_inliers=config.reseed_from_inliers,
    )
    pipe.load_observations()

    # Step A/B/C init
    points3d_init, best_pair_idx, inlier_mask = pipe.init_by_pair_selection()

    # BA refine using inlier views
    points3d_ref = pipe.refine_lm_per_point(points3d_init, inlier_mask)

    # 4) reprojection/errors (keep your key naming)
    proj_init, mean_init = pipe.per_cam_reprojection(points3d_init)
    proj_ref, mean_ref = pipe.per_cam_reprojection(points3d_ref)
    err_init = pipe.per_cam_errors(points3d_init, proj_init)
    err_ref = pipe.per_cam_errors(points3d_ref, proj_ref)

    pipe.save_npz(
        out_npz=out_npz,
        points3d_init=points3d_init,
        points3d_ref=points3d_ref,
        best_pair_idx=best_pair_idx,
        inlier_mask=inlier_mask,
        mean_init=mean_init,
        mean_ref=mean_ref,
        proj_init=proj_init,
        proj_ref=proj_ref,
        err_init=err_init,
        err_ref=err_ref,
        toml_sections=toml_sections,
    )

    print(f"[OK] Saved: {out_npz}")
    print("Mean reprojection error (px):")
    for cam in cams:
        print(f"  {cam}: init={mean_init[cam]:.2f}  refined={mean_ref[cam]:.2f}")

    # 5) debug disk outputs (same behavior)
    if debug_dir is not None:
        for cam in cams:
            cm = cameras[cam]
            w_img, h_img = cm.w, cm.h
            if img_dir is not None:
                img = read_image(img_dir, cam, fallback_size_wh=(w_img, h_img))
            else:
                img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

            obs = pipe.obs_per_cam[cam]
            over_init = draw_overlay(img, obs, proj_init[cam], subset.edges)
            cv2.putText(over_init, f"{cam} init mean_err={mean_init[cam]:.2f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(debug_dir / f"{cam}_overlay_init.jpg"), over_init)

            over_ref = draw_overlay(img, obs, proj_ref[cam], subset.edges)
            cv2.putText(over_ref, f"{cam} refined mean_err={mean_ref[cam]:.2f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(debug_dir / f"{cam}_overlay_refined.jpg"), over_ref)

        # Save a 3D scatter PNG (refined)
        try:
            import matplotlib.pyplot as plt
            X = points3d_ref
            ok = np.isfinite(X).all(axis=1)
            X = X[ok]
            if X.shape[0] > 0:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=12)
                ax.set_title("Triangulated 3D (refined)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                fig.tight_layout()
                fig.savefig(str(debug_dir / "triangulated_3d_refined.png"), dpi=160)
                plt.close(fig)
        except Exception as e:
            print(f"[WARN] Could not save 3D scatter PNG: {e}")

        print(f"[DEBUG] Saved debug files in: {debug_dir}")

    # 6) interactive 3D only
    if config.debug:
        show_3d_scatter_interactive(points3d_ref, title="Triangulated 3D (refined)")

    return TriangulationRunResult(
        out_npz=out_npz,
        points3d_init=points3d_init,
        points3d_refined=points3d_ref,
        mean_err_init=mean_init,
        mean_err_refined=mean_ref,
        debug_dir=debug_dir,
    )


def main(args: Optional[argparse.Namespace] = None) -> TriangulationRunResult:
    """CLI/programmatic entrypoint for stage-2 triangulation."""

    if args is None:
        args = parse_args()
    config = namespace_to_config(args)
    return run_triangulation(config)


if __name__ == "__main__":
    main()
