"""Shared utilities for datagen analytics scripts."""

from __future__ import annotations

import math
import os
import statistics
from pathlib import Path
from typing import Iterable

DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "data" / "selected_33w"
DEFAULT_RENDERS_DIR = Path(__file__).resolve().parent / "data" / "path_video_frames_random_humans_33w"
DEFAULT_ASSIGNMENTS_JSON = Path(__file__).resolve().parent / "data" / "actor_assignments_w_ban_33w.json"
FRAME_STEP_WORLD = 0.05
FRAME_THRESHOLDS = [30, 60, 90]

_MPL_CACHE = Path(__file__).resolve().parent / ".matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))


def resolve_label_directory(scene_dir: Path) -> Path | None:
    """Locate the directory that stores label JSON definitions for a scene."""

    label_paths_dir = scene_dir / "label_paths"
    if label_paths_dir.is_dir():
        return label_paths_dir
    if scene_dir.is_dir() and any(scene_dir.glob("*.json")):
        return scene_dir
    return None


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(sorted_vals[int(pos)])
    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    fraction = pos - lower
    return float(lower_val + (upper_val - lower_val) * fraction)


def describe(values: Iterable[float]) -> dict:
    seq = [float(v) for v in values if v is not None]
    if not seq:
        return {"count": 0}
    sorted_seq = sorted(seq)
    result = {
        "count": len(sorted_seq),
        "min": float(sorted_seq[0]),
        "max": float(sorted_seq[-1]),
        "mean": statistics.fmean(sorted_seq),
        "median": statistics.median(sorted_seq),
        "p10": _percentile(sorted_seq, 0.10),
        "p90": _percentile(sorted_seq, 0.90),
    }
    if len(sorted_seq) > 1:
        result["stddev"] = statistics.pstdev(sorted_seq)
    return result
