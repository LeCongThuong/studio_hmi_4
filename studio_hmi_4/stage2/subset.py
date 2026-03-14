"""Subset-selection helpers for the MHR-70 triangulation stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .io import import_py_module


@dataclass
class MHRSubset:
    subset_names: np.ndarray
    subset_indices: np.ndarray
    edges: List[Tuple[int, int]]


class MHRSubsetSelector:
    """Build name->index mapping and the triangulated hand+arm+torso subset."""

    def __init__(self, mhr_py: str):
        mod = import_py_module(mhr_py)
        pose_info = getattr(mod, "pose_info", None)
        if pose_info is None:
            raise AttributeError("mhr_70.py must define `pose_info` dict")
        self.pose_info: Dict = pose_info
        self.name_to_idx: Dict[str, int] = self._build_name_to_idx(pose_info)

    @staticmethod
    def _build_name_to_idx(pose_info: Dict) -> Dict[str, int]:
        name_to_idx: Dict[str, int] = {}

        keypoint_info = pose_info.get("keypoint_info", {})
        if isinstance(keypoint_info, dict) and len(keypoint_info) > 0:
            for idx_key, value in keypoint_info.items():
                idx = int(idx_key)
                if isinstance(value, dict) and "name" in value:
                    name_to_idx[str(value["name"])] = idx

        original = pose_info.get("original_keypoint_info", {})
        if isinstance(original, dict):
            for idx, name in original.items():
                name_to_idx[str(name)] = int(idx)

        return name_to_idx

    def build_subset(self) -> MHRSubset:
        left_hand = self.pose_info.get("left_hand_keypoint_names", [])
        right_hand = self.pose_info.get("right_hand_keypoint_names", [])
        must = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "neck",
            "left_acromion",
            "right_acromion",
        ]
        subset_names = list(right_hand) + list(left_hand) + must

        idxs: List[int] = []
        for name in subset_names:
            if name not in self.name_to_idx:
                sample = sorted(list(self.name_to_idx.keys()))[:40]
                raise KeyError(f"Keypoint name not found: {name}\nExample available names: {sample} ...")
            idxs.append(int(self.name_to_idx[name]))

        subset_indices = np.array(idxs, dtype=np.int32)
        return MHRSubset(
            subset_names=np.array(subset_names, dtype=object),
            subset_indices=subset_indices,
            edges=self._build_subset_edges(subset_indices),
        )

    def _build_subset_edges(self, subset_indices: np.ndarray) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        skeleton = self.pose_info.get("skeleton_info", None)
        if not isinstance(skeleton, dict):
            return edges

        subset_set = set(subset_indices.tolist())
        full_to_sub = {int(full): i for i, full in enumerate(subset_indices.tolist())}

        for _, edge in skeleton.items():
            a_name, b_name = edge["link"]
            if (a_name in self.name_to_idx) and (b_name in self.name_to_idx):
                ia = int(self.name_to_idx[a_name])
                ib = int(self.name_to_idx[b_name])
                if ia in subset_set and ib in subset_set:
                    edges.append((full_to_sub[ia], full_to_sub[ib]))
        return edges
