from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

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
    free_pixel_min: int = 250  # occupancy pixels considered free (>= threshold if free_is_white else <= threshold)
    free_is_white: bool = True  # True: free if >= threshold; False: free if <= threshold
    coverage_mode: Literal["area", "angular"] = "angular"  # "area" = wedge area, "angular" = FOV angular coverage
    desired_count: int | None = None  # guiding count
    priority: Literal["coverage", "count"] = "coverage"  # which requirement is treated as hard
    zone_weights: tuple[float, float, float] = (1.0, 2.0, 1.0)  # near:mid:far count ratios (soft)


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


def load_free_space_mask(dataset_dir: Path, *, threshold: int = 128, free_is_white: bool = True) -> tuple[np.ndarray, dict]:
    """
    Load occupancy.png and return a boolean free-space mask plus metadata.
    If free_is_white: free if grayscale >= threshold.
    If not free_is_white: free if grayscale <= threshold.
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
    free = mask >= threshold if free_is_white else mask <= threshold
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

def _disc_angular_width(radius: float, distance: float) -> float:
    """
    Angular span (radians) of a disc of radius r at range d, seen from origin.
    Formula: theta = 2 * asin(r / d). Small-angle approx: ~2r/d.
    """
    d = max(distance, radius + EPS)
    ratio = min(1.0, max(0.0, radius / d))
    return 2.0 * math.asin(ratio)


def _angle_between(forward: np.ndarray, vec: np.ndarray) -> float:
    """Signed angle between forward and vec in radians."""
    f = _normalize(forward)
    v = _normalize(vec)
    dot = float(np.clip(np.dot(f, v), -1.0, 1.0))
    det = float(f[0] * v[1] - f[1] * v[0])
    return math.atan2(det, dot)


def _effective_disc_span(radius: float, distance: float, fov_half: float, center_angle: float) -> float:
    """
    Limit disc angular span by the remaining FOV if the center is near the edge.
    """
    raw = _disc_angular_width(radius, distance)
    available_half = max(0.0, fov_half - abs(center_angle))
    return min(raw, 2.0 * available_half)


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
    if region.dtype != bool:
        region = region >= free_pixel_min
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
    coverage_mode: str,
    fov_rad: float,
    r_min: float,
    r_max: float,
    radius: float,
) -> int:
    if coverage_mode == "area":
        if disc_area <= 0.0 or wedge_area <= 0.0:
            return 0
    else:
        if fov_rad <= 0.0:
            return 0
    target = 0
    if target_coverage is not None and target_coverage > 0.0:
        clamped = min(1.0, max(0.0, target_coverage))
        if coverage_mode == "area":
            target = max(1, math.ceil(clamped * wedge_area / disc_area))
        else:
            # Conservative: use farthest distance to estimate smallest angular span.
            r_use = max(r_min, r_max, radius + EPS)
            ang = _disc_angular_width(radius, r_use)
            if ang > 0.0:
                target = max(1, math.ceil(clamped * fov_rad / ang))
    if max_npcs is not None and max_npcs > 0:
        target = max_npcs if target == 0 else min(target, max_npcs)
    return target


def _zone_bounds(r_min: float, r_max: float) -> list[tuple[float, float]]:
    """Split radial band into 3 equal spans (near, mid, far)."""
    if r_max <= r_min:
        return [(r_min, r_min), (r_min, r_min), (r_min, r_min)]
    span = r_max - r_min
    step = span / 3.0
    return [
        (r_min, r_min + step),
        (r_min + step, r_min + 2.0 * step),
        (r_min + 2.0 * step, r_max),
    ]


def _distribute_counts(total: int, weights: Sequence[float]) -> list[int]:
    """Allocate counts across zones based on weights, preserving total."""
    if total <= 0:
        return [0, 0, 0]
    w = np.array(weights, dtype=np.float32)
    w = np.where(w < 0.0, 0.0, w)
    if np.allclose(w.sum(), 0.0):
        w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    w = w / w.sum()
    raw = w * float(total)
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        frac = raw - base
        order = np.argsort(-frac)
        for idx in range(remainder):
            base[order[idx % len(base)]] += 1
    return base.tolist()


def plan_npc_positions(
    *,
    wedge: CameraWedge,
    free_mask: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    config: NPCDensityConfig,
    goal_xy: np.ndarray | None = None,
    exclude_discs: Sequence[tuple[np.ndarray, float]] | None = None,
) -> NPCPlacementResult:
    """
    Sample NPC positions inside the camera wedge with disc clearance.

    The sampler targets a coverage fraction (if provided) and falls back to max_npcs as a cap.
    It rejects candidates that collide with obstacles/other NPCs, fall outside the mask, or
    block the camera->goal segment when blocking is disabled. Optional exclude_discs are treated
    as occupied (collide but do not contribute to coverage).
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
    coverage_cap: int | None = None
    if config.target_coverage is not None and config.target_coverage > 0.0:
        cov = min(1.0, max(0.0, config.target_coverage))
        if config.coverage_mode == "area":
            coverage_cap = int(math.floor(cov * wedge_area / disc_area)) if wedge_area > 0.0 and disc_area > 0.0 else 0
        else:
            ang = _disc_angular_width(config.clearance_radius, max(r_max, config.clearance_radius + EPS))
            ang = min(ang, fov_rad)
            coverage_cap = int(math.floor(cov * fov_rad / ang)) if ang > 0.0 and fov_rad > 0.0 else 0
        coverage_cap = max(0, coverage_cap)

    desired = config.desired_count if (config.desired_count is not None and config.desired_count > 0) else None
    target_count = 0
    if config.priority == "count":
        if desired is not None:
            target_count = desired
        elif coverage_cap is not None:
            target_count = coverage_cap
    else:  # coverage priority
        if coverage_cap is not None:
            target_count = coverage_cap
        elif desired is not None:
            target_count = desired

    if coverage_cap is not None and target_count > coverage_cap:
        target_count = coverage_cap
    if config.max_npcs is not None and config.max_npcs > 0:
        target_count = min(target_count, config.max_npcs)

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

    zones = _zone_bounds(r_min, r_max)
    effective_weights = config.zone_weights if target_count >= 12 else (1.0, 1.0, 1.0)
    zone_counts = _distribute_counts(target_count, effective_weights)

    placements: list[np.ndarray] = []
    rejected_blocking = rejected_clearance = rejected_oob = 0
    attempts = 0
    max_attempts = max(config.max_resamples * max(target_count, 1), config.max_resamples)

    for zone_idx, need in enumerate(zone_counts):
        z_min, z_max = zones[zone_idx]
        while need > 0 and attempts < max_attempts:
            attempts += 1
            angle = rng.uniform(-0.5 * fov_rad, 0.5 * fov_rad)
            r = math.sqrt(rng.uniform(z_min * z_min, z_max * z_max)) if z_max > z_min else z_min
            direction = _rotate(forward_xy, angle)
            candidate = wedge.origin_xy + direction * r

            if not _inside_mask(meta, free_mask, candidate, config.clearance_radius, config.free_pixel_min):
                rejected_oob += 1
                continue

            too_close = any(
                float(np.linalg.norm(candidate - placed)) < 2.0 * config.clearance_radius - EPS
                for placed in placements
            )
            if not too_close and exclude_discs:
                for center, radius in exclude_discs:
                    if float(np.linalg.norm(candidate - center)) < (radius + config.clearance_radius - EPS):
                        too_close = True
                        break
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
            need -= 1

    achieved_coverage = 0.0
    if placements:
        if config.coverage_mode == "area":
            if wedge_area > 0.0:
                achieved_coverage = min(1.0, len(placements) * disc_area / wedge_area)
        else:
            total_angle = 0.0
            fov_half = 0.5 * fov_rad
            for pos in placements:
                offset = pos - wedge.origin_xy
                dist = float(np.linalg.norm(offset))
                center_angle = _angle_between(forward_xy, offset)
                span = _effective_disc_span(
                    config.clearance_radius,
                    max(dist, config.clearance_radius + EPS),
                    fov_half,
                    center_angle,
                )
                total_angle += span
            if fov_rad > 0.0:
                achieved_coverage = min(1.0, total_angle / fov_rad)
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
