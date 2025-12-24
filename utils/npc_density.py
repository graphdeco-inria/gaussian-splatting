from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import numpy as np

from utils.render_utils import load_occupancy_metadata, world_to_pixel

EPS = 1e-6


@dataclass(frozen=True)
class CameraWedge:
    """2D ground-plane slice of the camera FOV."""

    origin_xy: np.ndarray
    forward_xy: np.ndarray
    fov_deg: float
    max_range: float
    min_range: float = 0.0


@dataclass(frozen=True)
class NPCDensityConfig:
    """Knobs for NPC placement inside a camera wedge."""

    clearance_radius: float = 0.30  # meters
    min_distance_from_camera: float = 1.0  # meters
    max_distance_from_camera: float | None = None  # meters (None => use wedge.max_range)
    target_coverage: float | None = None  # fraction of wedge area to occupy (0..1)
    max_npcs: int | None = None  # hard cap regardless of coverage target
    allow_blocking: bool = False  # if False, avoid blocking camera->goal line
    max_resamples: int = 50  # per requested NPC
    free_pixel_min: int = 128  # occupancy pixels >= threshold are considered free


@dataclass(frozen=True)
class NPCPlacementResult:
    positions_xy: list[np.ndarray]
    requested_count: int
    achieved_coverage: float
    target_coverage: float | None
    attempts: int
    rejected_blocking: int
    rejected_clearance: int
    rejected_oob: int
    shortfall: int


def load_free_space_mask(dataset_dir: Path, *, threshold: int = 128) -> tuple[np.ndarray, dict]:
    """
    Load occupancy.png and return a boolean free-space mask plus metadata.
    A pixel is considered free if its grayscale value >= threshold.
    """
    meta = load_occupancy_metadata(dataset_dir)
    occ_png = dataset_dir / "occupancy.png"
    if not occ_png.is_file():
        raise FileNotFoundError(f"Missing occupancy.png in {dataset_dir}")

    mask = imageio.imread(occ_png)
    if mask.ndim == 3:
        # Convert RGB occupancy to luma
        mask = np.round(
            0.2126 * mask[..., 0] + 0.7152 * mask[..., 1] + 0.0722 * mask[..., 2]
        ).astype(np.uint8)
    free = mask >= threshold
    return free.astype(bool), meta


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        raise ValueError("Cannot normalise zero-length vector.")
    return (vec / norm).astype(np.float32)


def _rotate(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y = float(vec[0]), float(vec[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def _wedge_area(fov_rad: float, r_min: float, r_max: float) -> float:
    if r_max <= r_min or fov_rad <= 0.0:
        return 0.0
    return 0.5 * fov_rad * (r_max * r_max - r_min * r_min)


def _disc_area(radius: float) -> float:
    if radius <= 0.0:
        return 0.0
    return math.pi * radius * radius


def _blocks_goal(camera_xy: np.ndarray, goal_xy: np.ndarray, candidate_xy: np.ndarray, radius: float) -> bool:
    """Return True if the candidate disc overlaps the camera->goal segment."""
    seg = goal_xy - camera_xy
    seg_len = float(np.linalg.norm(seg))
    if seg_len < EPS:
        return False
    t = float(np.dot(candidate_xy - camera_xy, seg)) / (seg_len * seg_len)
    if t < 0.0 or t > 1.0:
        return False
    closest = camera_xy + t * seg
    return float(np.linalg.norm(candidate_xy - closest)) <= radius


def _inside_mask(meta: dict, free_mask: np.ndarray, xy: np.ndarray, radius_m: float, free_pixel_min: int) -> bool:
    """Check if xy is in-bounds and clears occupancy (using a circular footprint)."""
    h, w = free_mask.shape[:2]
    u, v = world_to_pixel(meta, xy)
    radius_px = int(math.ceil(radius_m / float(meta["scale"])))
    if u < radius_px or v < radius_px or u >= w - radius_px or v >= h - radius_px:
        return False
    # Build a circular mask to test clearance
    uu = np.arange(u - radius_px, u + radius_px + 1)
    vv = np.arange(v - radius_px, v + radius_px + 1)
    du = uu[np.newaxis, :] - u
    dv = vv[:, np.newaxis] - v
    circle = (du * du + dv * dv) <= (radius_px * radius_px)
    region = free_mask[v - radius_px : v + radius_px + 1, u - radius_px : u + radius_px + 1]
    if region.shape != circle.shape:
        return False
    # Occupancy is free if all covered pixels are above the threshold
    return bool(np.all(region[circle]))


def _estimate_target_count(
    *,
    wedge_area: float,
    disc_area: float,
    target_coverage: float | None,
    max_npcs: int | None,
) -> int:
    if disc_area <= 0.0 or wedge_area <= 0.0:
        return 0
    target = 0
    if target_coverage is not None and target_coverage > 0.0:
        target = max(1, math.ceil(target_coverage * wedge_area / disc_area))
    if max_npcs is not None and max_npcs > 0:
        target = max(1 if target == 0 else min(target, max_npcs), 0)
    return target


def plan_npc_positions(
    *,
    wedge: CameraWedge,
    free_mask: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    config: NPCDensityConfig,
    goal_xy: np.ndarray | None = None,
) -> NPCPlacementResult:
    """
    Sample NPC positions inside the camera wedge with disc clearance.

    The sampler targets a coverage fraction (if provided) and falls back to max_npcs as a cap.
    It rejects candidates that collide with obstacles/other NPCs, fall outside the mask, or
    block the camera->goal segment when blocking is disabled.
    """
    forward_xy = _normalize(wedge.forward_xy)
    fov_rad = math.radians(wedge.fov_deg)
    r_min = max(config.min_distance_from_camera, wedge.min_range)
    r_max = wedge.max_range if config.max_distance_from_camera is None else min(wedge.max_range, config.max_distance_from_camera)
    if r_max <= r_min:
        return NPCPlacementResult(
            positions_xy=[],
            requested_count=0,
            achieved_coverage=0.0,
            target_coverage=config.target_coverage,
            attempts=0,
            rejected_blocking=0,
            rejected_clearance=0,
            rejected_oob=0,
            shortfall=0,
        )

    wedge_area = _wedge_area(fov_rad, r_min, r_max)
    disc_area = _disc_area(config.clearance_radius)
    target_count = _estimate_target_count(
        wedge_area=wedge_area,
        disc_area=disc_area,
        target_coverage=config.target_coverage,
        max_npcs=config.max_npcs,
    )
    if target_count <= 0:
        return NPCPlacementResult(
            positions_xy=[],
            requested_count=0,
            achieved_coverage=0.0,
            target_coverage=config.target_coverage,
            attempts=0,
            rejected_blocking=0,
            rejected_clearance=0,
            rejected_oob=0,
            shortfall=0,
        )

    placements: list[np.ndarray] = []
    rejected_blocking = rejected_clearance = rejected_oob = 0
    attempts = 0
    max_attempts = max(config.max_resamples * target_count, config.max_resamples)

    while len(placements) < target_count and attempts < max_attempts:
        attempts += 1
        angle = rng.uniform(-0.5 * fov_rad, 0.5 * fov_rad)
        r = math.sqrt(rng.uniform(r_min * r_min, r_max * r_max))
        direction = _rotate(forward_xy, angle)
        candidate = wedge.origin_xy + direction * r

        if not _inside_mask(meta, free_mask, candidate, config.clearance_radius, config.free_pixel_min):
            rejected_oob += 1
            continue

        too_close = any(
            float(np.linalg.norm(candidate - placed)) < 2.0 * config.clearance_radius - EPS
            for placed in placements
        )
        if too_close:
            rejected_clearance += 1
            continue

        if (
            not config.allow_blocking
            and goal_xy is not None
            and _blocks_goal(wedge.origin_xy, goal_xy, candidate, config.clearance_radius)
        ):
            rejected_blocking += 1
            continue

        placements.append(candidate.astype(np.float32))

    achieved_coverage = 0.0
    if wedge_area > 0.0 and placements:
        achieved_coverage = min(1.0, len(placements) * disc_area / wedge_area)
    shortfall = max(0, target_count - len(placements))

    return NPCPlacementResult(
        positions_xy=placements,
        requested_count=target_count,
        achieved_coverage=achieved_coverage,
        target_coverage=config.target_coverage,
        attempts=attempts,
        rejected_blocking=rejected_blocking,
        rejected_clearance=rejected_clearance,
        rejected_oob=rejected_oob,
        shortfall=shortfall,
    )


def estimate_coverage_for_positions(
    positions_xy: Sequence[np.ndarray],
    *,
    wedge: CameraWedge,
    config: NPCDensityConfig,
) -> float:
    """Compute coverage achieved by already-placed NPCs."""
    wedge_area = _wedge_area(math.radians(wedge.fov_deg), max(wedge.min_range, config.min_distance_from_camera), wedge.max_range)
    disc_area = _disc_area(config.clearance_radius)
    if wedge_area <= 0.0 or disc_area <= 0.0:
        return 0.0
    return min(1.0, len(list(positions_xy)) * disc_area / wedge_area)
