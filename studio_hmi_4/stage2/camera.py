"""Camera-rig parsing and projection helpers for stage-2 triangulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from studio_hmi_4.common.cv2_compat import require_cv2

try:
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class CameraModel:
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
        cv2 = require_cv2("stage-2 camera model construction")
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
        P_norm = np.hstack([R, tvec.reshape(3, 1)])
        return CameraModel(
            name=name,
            section=section,
            K=K,
            D=D,
            rvec=rvec,
            tvec=tvec,
            P_norm=P_norm,
            w=w_img,
            h=h_img,
        )

    @staticmethod
    def _invert_extrinsics(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cv2 = require_cv2("stage-2 camera extrinsic inversion")
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        t_inv = -R_inv @ tvec
        rvec_inv, _ = cv2.Rodrigues(R_inv)
        return rvec_inv.reshape(3, 1), t_inv.reshape(3, 1)

    def undistort_to_normalized(self, pts_px: np.ndarray) -> np.ndarray:
        cv2 = require_cv2("stage-2 keypoint undistortion")
        pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
        und = cv2.undistortPoints(pts, self.K, self.D)
        return und.reshape(-1, 2)

    def project_points_px(self, Xs_world: np.ndarray) -> np.ndarray:
        cv2 = require_cv2("stage-2 projection")
        Xw = Xs_world.reshape(-1, 1, 3).astype(np.float64)
        uv, _ = cv2.projectPoints(Xw, self.rvec, self.tvec, self.K, self.D)
        return uv.reshape(-1, 2).astype(np.float64)

    def project_point_px(self, X_world: np.ndarray) -> np.ndarray:
        cv2 = require_cv2("stage-2 projection")
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
                name=cam_name,
                section=section,
                block=block,
                invert_extrinsics=invert_extrinsics,
            )
        return out
