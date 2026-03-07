"""Frame discovery helpers for the full sequence pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from studio_hmi_4.stage2.runner import NP_EXTS, find_existing_with_exts

from .types import FrameInputEntry, FullPipelineConfig


def rel_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "" if str(rel) == "." else rel.as_posix()


def parse_frame_index(rel_dir: str) -> Optional[int]:
    if rel_dir == "":
        return None
    leaf = rel_dir.replace("\\", "/").split("/")[-1]
    return int(leaf) if leaf.isdigit() else None


def frame_sort_key(rel_dir: str):
    if rel_dir == "":
        return [(0, "")]
    toks = rel_dir.replace("\\", "/").split("/")
    out = []
    for token in toks:
        if token.isdigit():
            out.append((0, f"{int(token):020d}"))
        else:
            out.append((1, token))
    return out


def available_cams_in_dir(dir_path: Path, cams: Sequence[str]) -> List[str]:
    return [cam for cam in cams if find_existing_with_exts(dir_path, cam, NP_EXTS) is not None]


def dir_has_min_cam_predictions(dir_path: Path, cams: Sequence[str], min_views: int) -> bool:
    return len(available_cams_in_dir(dir_path, cams)) >= int(min_views)


def discover_frame_inputs(
    npy_root: Path,
    cams: Sequence[str],
    frame_rel: Optional[str] = None,
) -> List[FrameInputEntry]:
    if frame_rel:
        target = (npy_root / frame_rel).resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"Requested frame_rel directory not found: {target}")
        rel_dir = rel_key(target, npy_root)
        return [
            FrameInputEntry(
                rel_dir=rel_dir,
                npy_dir=target,
                frame_index=parse_frame_index(rel_dir),
                available_cams=available_cams_in_dir(target, cams),
            )
        ]

    candidates = [npy_root]
    candidates.extend(sorted(p for p in npy_root.rglob("*") if p.is_dir()))
    entries: List[FrameInputEntry] = []
    for path in candidates:
        available = available_cams_in_dir(path, cams)
        if len(available) == 0:
            continue
        rel_dir = rel_key(path, npy_root)
        entries.append(
            FrameInputEntry(
                rel_dir=rel_dir,
                npy_dir=path,
                frame_index=parse_frame_index(rel_dir),
                available_cams=available,
            )
        )
    entries.sort(key=lambda entry: frame_sort_key(entry.rel_dir))
    return entries


def inject_numeric_gaps(entries: List[FrameInputEntry]) -> List[FrameInputEntry]:
    if len(entries) < 2:
        return entries
    if any(entry.frame_index is None for entry in entries):
        return entries
    if any("/" in entry.rel_dir for entry in entries if entry.rel_dir != ""):
        return entries

    by_idx = {int(entry.frame_index): entry for entry in entries if entry.frame_index is not None}
    all_idx = sorted(by_idx.keys())
    out: List[FrameInputEntry] = []
    for idx in range(all_idx[0], all_idx[-1] + 1):
        if idx in by_idx:
            out.append(by_idx[idx])
        else:
            out.append(
                FrameInputEntry(
                    rel_dir=str(idx),
                    npy_dir=None,
                    frame_index=idx,
                    available_cams=[],
                )
            )
    return out


def cam_to_section_map(cams: Sequence[str], toml_sections: Optional[Sequence[str]]) -> Dict[str, str]:
    if toml_sections is None or len(toml_sections) == 0:
        return {cam: cam for cam in cams}
    if len(toml_sections) != len(cams):
        raise ValueError("--toml_sections must match length of --cams (or omit).")
    return {cam: section for cam, section in zip(cams, toml_sections)}


def expected_stage1_meta(config: FullPipelineConfig, image_root: Path) -> Dict[str, Any]:
    include_rel_dirs = [config.frame_rel] if config.frame_rel else None
    return {
        "image_folder": str(image_root.resolve()),
        "include_rel_dirs": include_rel_dirs,
        "checkpoint_path": str(config.checkpoint_path),
        "detector_name": str(config.detector_name),
        "segmentor_name": str(config.segmentor_name),
        "fov_name": str(config.fov_name),
        "person_select_strategy": str(config.person_select_strategy),
        "person_index": int(config.person_index),
        "enable_specialized_hand_fusion": bool(config.enable_specialized_hand_fusion),
        "specialized_hand_source": str(config.specialized_hand_source),
        "specialized_hand_model": str(config.specialized_hand_model),
        "specialized_hand_input_root": (
            None
            if str(config.specialized_hand_input_root).strip() == ""
            else str(Path(config.specialized_hand_input_root).expanduser().resolve())
        ),
        "specialized_hand_detector_conf": float(config.specialized_hand_detector_conf),
        "specialized_hand_rescale_factor": float(config.specialized_hand_rescale_factor),
        "specialized_hand_wrist_max_dist_px": float(config.specialized_hand_wrist_max_dist_px),
        "replace_wrist_with_specialized": bool(config.replace_wrist_with_specialized),
        "specialized_hand_debug_vis": bool(config.specialized_hand_debug_vis),
        "specialized_hand_debug_dirname": str(config.specialized_hand_debug_dirname),
        "wilor_repo_id": str(config.wilor_repo_id),
    }
