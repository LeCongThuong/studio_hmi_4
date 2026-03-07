"""Summary/config persistence helpers for sequence orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .types import FramePipelineResult, FullPipelineConfig


def save_summaries(
    output_root: Path,
    npy_root: Path,
    frame_results: List[FramePipelineResult],
) -> Path:
    status_counts: Dict[str, int] = {}
    for fr in frame_results:
        status_counts[fr.status] = status_counts.get(fr.status, 0) + 1

    payload = {
        "output_root": str(output_root),
        "npy_root": str(npy_root),
        "num_frames": len(frame_results),
        "status_counts": status_counts,
        "frames": [
            {
                "rel_dir": fr.rel_dir,
                "frame_index": fr.frame_index,
                "npy_dir": None if fr.npy_dir is None else str(fr.npy_dir),
                "available_cams": fr.available_cams,
                "used_cams": fr.used_cams,
                "triangulated_npz": None if fr.triangulated_npz is None else str(fr.triangulated_npz),
                "optimized_npy": None if fr.optimized_npy is None else str(fr.optimized_npy),
                "smoothed_npy": None if fr.smoothed_npy is None else str(fr.smoothed_npy),
                "status": fr.status,
                "best_loss": fr.best_loss,
                "final_loss": fr.final_loss,
                "best_data_loss": fr.best_data_loss,
                "final_data_loss": fr.final_data_loss,
                "best_iter": fr.best_iter,
                "is_bad_loss": fr.is_bad_loss,
                "recovered_from": fr.recovered_from,
                "error": fr.error,
            }
            for fr in frame_results
        ],
    }
    out = (output_root / "sequence_summary.json").resolve()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def save_resolved_config(output_root: Path, config: FullPipelineConfig) -> Path:
    out = (output_root / "pipeline_config.json").resolve()
    out.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return out
