#!/usr/bin/env python3
"""Probe which MHR-133 body-pose indices drive which named MHR-70 keypoints.

Usage:
  python3 inspect_body_pose_param_effects.py \
    --npy /path/to/stage1/front.npy \
    --ckpt ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_pt ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --device cuda \
    --delta 0.05 \
    --target_group arm \
    --out_csv body_pose_param_effects.csv \
    --out_md body_pose_param_effects.md

This script:
1. loads one saved MHR parameter bundle (`body_pose_params`, `hand_pose_params`, `scale_params`, `shape_params`, `expr_params`)
2. perturbs one `body_pose_params[i]` at a time by +/-delta
3. runs MHR forward through the repo's stage-3 runtime
4. measures which of the 70 named MHR keypoints move the most
5. prints and optionally saves a compact report
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from mhr70 import pose_info
from studio_hmi_4.common import load_npy_dict
from studio_hmi_4.stage3.alignment import MHR_PARAM_HAND_IDXS_133
from studio_hmi_4.stage3.runtime import (
    apply_repo_camera_flip_xyz,
    get_param_array,
    load_sam_head,
    mhr_fk,
    to_torch,
)
from studio_hmi_4.stage3.types import OptimizationConfig


KEYPOINT_INFO = pose_info["keypoint_info"]
KEYPOINT_NAMES = [str(KEYPOINT_INFO[i]["name"]) for i in range(len(KEYPOINT_INFO))]
LEFT_HAND_NAMES = {str(name) for name in pose_info.get("left_hand_keypoint_names", [])}
RIGHT_HAND_NAMES = {str(name) for name in pose_info.get("right_hand_keypoint_names", [])}
FOOT_NAMES = {str(name) for name in pose_info.get("foot_keypoint_names", [])}
BODY_NAMES = {str(name) for name in pose_info.get("body_keypoint_names", [])}
LOWER_BODY_NAMES = {name for name in BODY_NAMES | FOOT_NAMES if any(tag in name for tag in ("hip", "knee", "ankle", "toe", "heel"))}
ARM_NAMES = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "left_wrist",
    "right_wrist",
}
ARM_AND_HAND_NAMES = ARM_NAMES | LEFT_HAND_NAMES | RIGHT_HAND_NAMES
HEAD_TORSO_NAMES = set(KEYPOINT_NAMES) - LOWER_BODY_NAMES - LEFT_HAND_NAMES - RIGHT_HAND_NAMES - ARM_NAMES


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", required=True, type=Path, help="Path to one saved Stage-1/Stage-3 .npy containing MHR params.")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hf_repo", type=str, default=None)
    grp.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mhr_pt", type=str, default="", help="Required with --ckpt.")

    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--delta", type=float, default=0.05, help="Symmetric perturbation magnitude for each body-pose index.")
    ap.add_argument("--indices", type=str, default="", help="Comma-separated indices/ranges, e.g. '0:31,124:130,62,90'. Default: all.")
    ap.add_argument("--target_group", choices=["arm", "lower"], default="arm", help="Which motion family to prioritize in the ranked summary.")
    ap.add_argument("--top_k", type=int, default=5, help="How many top-moving keypoints to report per perturbed index.")
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional CSV report path.")
    ap.add_argument("--out_md", type=Path, default=None, help="Optional Markdown summary path.")
    ap.add_argument("--out_json", type=Path, default=None, help="Optional JSON summary path with recommended index lists.")
    ap.add_argument("--limit_print", type=int, default=20, help="How many candidate rows to print in the terminal summary.")
    ap.add_argument("--recommend_top_n", type=int, default=16, help="How many top-ranked indices to include in the recommended index lists.")
    return ap


def parse_indices(spec: str, pose_dim: int) -> List[int]:
    if not spec.strip():
        return list(range(int(pose_dim)))
    out: List[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            bits = token.split(":")
            if len(bits) != 2:
                raise ValueError(f"Bad range token '{token}'. Expected start:end.")
            start = int(bits[0])
            end = int(bits[1])
            out.extend(range(start, end))
        else:
            out.append(int(token))
    out = sorted({idx for idx in out if 0 <= idx < int(pose_dim)})
    if not out:
        raise ValueError("No valid pose indices selected after parsing --indices.")
    return out


def group_mean_mm(motion_mm: np.ndarray, names: Sequence[str], allowed: Iterable[str]) -> float:
    allowed_set = set(allowed)
    vals = [float(motion_mm[i]) for i, name in enumerate(names) if name in allowed_set]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def dominant_group(row: Dict[str, float]) -> str:
    scores = {
        "lower": float(row["lower_mean_mm"]),
        "arm": float(row["arm_mean_mm"]),
        "left_hand": float(row["left_hand_mean_mm"]),
        "right_hand": float(row["right_hand_mean_mm"]),
        "head_torso": float(row["head_torso_mean_mm"]),
    }
    return max(scores, key=scores.get)


def heuristic_label(top_names: Sequence[str]) -> str:
    top_set = set(top_names[: min(6, len(top_names))])
    if top_set & {"left_shoulder", "right_shoulder", "left_acromion", "right_acromion"}:
        return "shoulder_chain"
    if top_set & {"left_elbow", "right_elbow", "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa"}:
        return "elbow_chain"
    if top_set & {"left_wrist", "right_wrist"}:
        return "wrist_chain"
    if top_set & (LEFT_HAND_NAMES | RIGHT_HAND_NAMES):
        return "hand_chain"
    if {"left_hip", "right_hip"}.issubset(top_set):
        return "pelvis_or_hip_root_like"
    if {"left_knee", "right_knee"} & top_set:
        return next(name for name in top_names if "knee" in name)
    if {"left_ankle", "right_ankle"} & top_set:
        return next(name for name in top_names if "ankle" in name)
    if {"left_hip", "right_hip"} & top_set:
        return next(name for name in top_names if "hip" in name)
    return top_names[0] if top_names else "unknown"


def arm_focus_metrics(row: Dict[str, object]) -> tuple[float, float]:
    arm_hand_mean = float(row["arm_hand_mean_mm"])
    lower_mean = float(row["lower_mean_mm"])
    head_torso_mean = float(row["head_torso_mean_mm"])
    non_arm_mean = max(0.0, 0.5 * (lower_mean + head_torso_mean))
    ratio = arm_hand_mean / max(non_arm_mean, 1e-6)
    score = arm_hand_mean * ratio
    return score, ratio


def select_candidate_rows(rows: Sequence[Dict[str, object]], target_group: str) -> List[Dict[str, object]]:
    if target_group == "lower":
        candidates = [row for row in rows if row["dominant_group"] == "lower"]
        candidates.sort(
            key=lambda row: (float(row["lower_mean_mm"]), float(row["max_move_mm"])),
            reverse=True,
        )
        return candidates

    candidates = [
        row
        for row in rows
        if row["dominant_group"] in {"arm", "left_hand", "right_hand"}
    ]
    if not candidates:
        candidates = list(rows)
    candidates.sort(
        key=lambda row: (
            float(row["arm_focus_score"]),
            float(row["arm_focus_ratio"]),
            float(row["arm_hand_mean_mm"]),
            float(row["max_move_mm"]),
        ),
        reverse=True,
    )
    return candidates


def recommended_indices(
    rows: Sequence[Dict[str, object]],
    target_group: str,
    top_n: int,
    pose_dim: int,
) -> tuple[List[int], List[int]]:
    ranked = select_candidate_rows(rows, target_group)
    top_rows = ranked[: max(1, int(top_n))]
    all_idxs = [int(row["pose_idx"]) for row in top_rows]
    hand_slots = {int(i) for i in np.asarray(MHR_PARAM_HAND_IDXS_133, dtype=np.int64).reshape(-1) if 0 <= int(i) < int(pose_dim)}
    tail_slots = {int(i) for i in range(max(0, int(pose_dim) - 3), int(pose_dim))}
    body_only = [idx for idx in all_idxs if idx not in hand_slots and idx not in tail_slots]
    return all_idxs, body_only


def forward_k70(
    head,
    device: torch.device,
    pose: torch.Tensor,
    hand: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Tensor,
    expr: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        out = mhr_fk(
            head,
            pose,
            hand,
            scale,
            shape,
            expr,
            device,
            want_verts=False,
            want_joint=False,
            want_model_params=False,
        )
        k70 = apply_repo_camera_flip_xyz(out[1].squeeze(0)[:70])
        return k70.detach().cpu().numpy().astype(np.float32)


def write_csv(rows: Sequence[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pose_idx",
        "dominant_group",
        "heuristic_label",
        "max_move_mm",
        "lower_mean_mm",
        "arm_mean_mm",
        "arm_hand_mean_mm",
        "arm_focus_ratio",
        "arm_focus_score",
        "left_hand_mean_mm",
        "right_hand_mean_mm",
        "head_torso_mean_mm",
        "top_names",
        "top_moves_mm",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_md(rows: Sequence[Dict[str, object]], out_md: Path, limit_print: int, args: argparse.Namespace, pose_dim: int) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows = select_candidate_rows(rows, args.target_group)
    rec_all, rec_body_only = recommended_indices(rows, args.target_group, args.recommend_top_n, pose_dim)
    if args.target_group == "arm":
        section_title = "Candidate Arm-Dominant Indices"
        value_key = "arm_focus_score"
        label_key = "arm_focus_score"
    else:
        section_title = "Candidate Lower-Body Indices"
        value_key = "lower_mean_mm"
        label_key = "lower_mean_mm"
    lines = [
        "# Body Pose Parameter Effects",
        "",
        f"- `npy`: `{args.npy}`",
        f"- `delta`: `{args.delta}`",
        f"- `device`: `{args.device}`",
        f"- `indices`: `{args.indices or 'all'}`",
        f"- `target_group`: `{args.target_group}`",
        "",
        f"## {section_title}",
        "",
        "| pose_idx | label | score | max_move_mm | top keypoints |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in candidate_rows[: max(1, int(limit_print))]:
        lines.append(
            f"| {row['pose_idx']} | {row['heuristic_label']} | "
            f"{float(row[value_key]):.3f} | {float(row['max_move_mm']):.3f} | {row['top_names']} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Indices",
            "",
            f"- all: `{rec_all}`",
            f"- body-only (exclude `MHR_PARAM_HAND_IDXS_133` and trailing 3 dims): `{rec_body_only}`",
            "",
            "```python",
            f"{args.target_group.upper()}_POSE_IDXS_133 = np.array({rec_body_only}, dtype=np.int64)",
            "```",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(
    rows: Sequence[Dict[str, object]],
    out_json: Path,
    args: argparse.Namespace,
    pose_dim: int,
) -> None:
    import json

    out_json.parent.mkdir(parents=True, exist_ok=True)
    rec_all, rec_body_only = recommended_indices(rows, args.target_group, args.recommend_top_n, pose_dim)
    payload = {
        "npy": str(args.npy),
        "delta": float(args.delta),
        "target_group": str(args.target_group),
        "recommend_top_n": int(args.recommend_top_n),
        "recommended_all_indices": rec_all,
        "recommended_body_only_indices": rec_body_only,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    npy_path = args.npy.expanduser().resolve()
    data = load_npy_dict(npy_path)
    if "body_pose_params" not in data:
        raise KeyError(f"Missing body_pose_params in: {npy_path}")

    cfg = OptimizationConfig(
        npz=Path("runtime.npz"),
        npy_dir=npy_path.parent,
        cams=["front"],
        out_npy=Path("runtime.npy"),
        hf_repo=args.hf_repo,
        ckpt=args.ckpt,
        mhr_pt=args.mhr_pt,
        device=args.device,
    )
    head = load_sam_head(cfg, args.device)
    device = torch.device(args.device)

    pose = to_torch(np.asarray(data["body_pose_params"], dtype=np.float32), device).flatten().to(torch.float32)
    hand = get_param_array(data, "hand_pose_params", device).to(torch.float32)
    scale = get_param_array(data, "scale_params", device).to(torch.float32)
    shape = get_param_array(data, "shape_params", device).to(torch.float32)
    expr = get_param_array(data, "expr_params", device).to(torch.float32)

    pose_dim = int(pose.numel())
    indices = parse_indices(args.indices, pose_dim)
    base_k70 = forward_k70(head, device, pose, hand, scale, shape, expr)

    rows: List[Dict[str, object]] = []
    print(f"[inspect] npy={npy_path}")
    print(f"[inspect] pose_dim={pose_dim} selected_indices={len(indices)} delta={args.delta}")
    if "expr_params" not in data:
        print("[inspect] expr_params missing in input; script is using a zero fallback for this probe.")

    for pose_idx in indices:
        pose_plus = pose.clone()
        pose_minus = pose.clone()
        pose_plus[pose_idx] += float(args.delta)
        pose_minus[pose_idx] -= float(args.delta)

        k70_plus = forward_k70(head, device, pose_plus, hand, scale, shape, expr)
        k70_minus = forward_k70(head, device, pose_minus, hand, scale, shape, expr)

        motion_plus = np.linalg.norm(k70_plus - base_k70, axis=1)
        motion_minus = np.linalg.norm(k70_minus - base_k70, axis=1)
        motion_mm = 1000.0 * 0.5 * (motion_plus + motion_minus)

        order = np.argsort(-motion_mm)
        top_idx = order[: max(1, int(args.top_k))]
        top_names = [KEYPOINT_NAMES[i] for i in top_idx]
        top_moves = [float(motion_mm[i]) for i in top_idx]

        row: Dict[str, object] = {
            "pose_idx": int(pose_idx),
            "max_move_mm": float(motion_mm[order[0]]),
            "lower_mean_mm": group_mean_mm(motion_mm, KEYPOINT_NAMES, LOWER_BODY_NAMES),
            "arm_mean_mm": group_mean_mm(motion_mm, KEYPOINT_NAMES, ARM_NAMES),
            "arm_hand_mean_mm": group_mean_mm(motion_mm, KEYPOINT_NAMES, ARM_AND_HAND_NAMES),
            "left_hand_mean_mm": group_mean_mm(motion_mm, KEYPOINT_NAMES, LEFT_HAND_NAMES),
            "right_hand_mean_mm": group_mean_mm(motion_mm, KEYPOINT_NAMES, RIGHT_HAND_NAMES),
            "head_torso_mean_mm": group_mean_mm(motion_mm, KEYPOINT_NAMES, HEAD_TORSO_NAMES),
            "top_names": ", ".join(top_names),
            "top_moves_mm": ", ".join(f"{v:.3f}" for v in top_moves),
        }
        row["dominant_group"] = dominant_group(row)  # type: ignore[arg-type]
        row["heuristic_label"] = heuristic_label(top_names)
        arm_focus_score, arm_focus_ratio = arm_focus_metrics(row)
        row["arm_focus_score"] = arm_focus_score
        row["arm_focus_ratio"] = arm_focus_ratio
        rows.append(row)

    candidate_rows = select_candidate_rows(rows, args.target_group)
    value_key = "arm_focus_score" if args.target_group == "arm" else "lower_mean_mm"
    rec_all, rec_body_only = recommended_indices(rows, args.target_group, args.recommend_top_n, pose_dim)

    print(f"\nCandidate {args.target_group}-dominant pose indices:")
    for row in candidate_rows[: max(1, int(args.limit_print))]:
        print(
            f"  idx={int(row['pose_idx']):3d} "
            f"label={row['heuristic_label']:<24s} "
            f"{value_key}={float(row[value_key]):9.3f} "
            f"max={float(row['max_move_mm']):7.3f} mm "
            f"top=[{row['top_names']}]"
        )

    print("\nRecommended indices:")
    print(f"  all={rec_all}")
    print(f"  body_only={rec_body_only}")
    print("  ready_to_paste_python:")
    print(f"    {args.target_group.upper()}_POSE_IDXS_133 = np.array({rec_body_only}, dtype=np.int64)")

    if args.out_csv is not None:
        out_csv = args.out_csv.expanduser().resolve()
        write_csv(rows, out_csv)
        print(f"\nSaved CSV: {out_csv}")
    if args.out_md is not None:
        out_md = args.out_md.expanduser().resolve()
        write_md(rows, out_md, args.limit_print, args, pose_dim)
        print(f"Saved Markdown summary: {out_md}")
    if args.out_json is not None:
        out_json = args.out_json.expanduser().resolve()
        write_json(rows, out_json, args, pose_dim)
        print(f"Saved JSON summary: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
