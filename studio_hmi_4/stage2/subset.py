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
    """Build name->index mapping and the triangulated MHR-70 subset."""

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
        keypoint_info = self.pose_info.get("keypoint_info", {})
        if not isinstance(keypoint_info, dict) or len(keypoint_info) == 0:
            raise AttributeError("pose_info['keypoint_info'] must be a non-empty dict.")

        ordered_items = sorted((int(idx), value) for idx, value in keypoint_info.items())
        subset_names = []
        idxs: List[int] = []
        for idx, value in ordered_items:
            if not isinstance(value, dict) or "name" not in value:
                raise KeyError(f"Keypoint entry {idx} is missing a 'name' field.")
            subset_names.append(str(value["name"]))
            idxs.append(int(idx))

        if len(set(idxs)) != len(idxs):
            sample = subset_names[:20]
            raise RuntimeError(
                "Full MHR subset contains duplicate indices. "
                f"subset_size={len(subset_names)} sample={sample}"
            )

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
