"""Shared sorting and relative-path helpers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def natural_tokens(text: str):
    parts = re.split(r"(\d+)", text.replace("\\", "/"))
    tokens = []
    for part in parts:
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.lower()))
    return tokens


def parse_leaf_index(rel_dir: str) -> Optional[int]:
    leaf = rel_dir.replace("\\", "/").strip("/").split("/")[-1] if rel_dir else ""
    return int(leaf) if leaf.isdigit() else None


def sort_key(rel_dir: str):
    idx = parse_leaf_index(rel_dir)
    if idx is not None:
        return (0, idx, rel_dir)
    return (1, natural_tokens(rel_dir), rel_dir)


def normalize_rel_dir(rel_dir: str) -> str:
    return rel_dir.replace("\\", "/").strip("/")


def relative_dir(path: str | Path, root: str | Path) -> str:
    rel = str(Path(path).resolve().parent.relative_to(Path(root).resolve()))
    return "" if rel == "." else rel
