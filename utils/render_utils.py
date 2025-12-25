from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix

EPS = 1e-6


def build_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Construct a right-handed look-at view matrix (forward = target - eye)."""

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < EPS:
        raise ValueError("Camera target too close to position; cannot build view matrix.")
    forward /= forward_norm

    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < EPS:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(fallback, forward)) > 0.99:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fallback, forward)
        right_norm = np.linalg.norm(right)
    right /= max(right_norm, EPS)

    true_up = np.cross(forward, right)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def build_perspective_camera(
    position: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    znear: float,
    zfar: float,
    device: torch.device,
) -> MiniCam:
    """Generate a MiniCam with perspective projection."""

    if height <= 0 or width <= 0:
        raise ValueError("Camera resolution must be positive.")

    fovy = math.radians(fov_deg)
    aspect = width / max(height, 1)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * aspect)

    view = build_look_at(position, target, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    world_view = torch.from_numpy(view).to(device).transpose(0, 1)
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).to(device).transpose(0, 1)
    full_proj = (world_view.unsqueeze(0) @ projection.unsqueeze(0)).squeeze(0)

    return MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
    )


def world_to_pixel(meta: dict, xy: np.ndarray) -> tuple[int, int]:
    """Convert world (x,y) to occupancy image pixel (u,v)."""

    x, y = float(xy[0]), float(xy[1])
    u = int(round((x - float(meta["left"])) / float(meta["scale"])))
    v = int(round((float(meta["top"]) - y) / float(meta["scale"])))
    return u, v


def pixel_to_world(meta: dict, uv: Tuple[int, int]) -> np.ndarray:
    """Convert occupancy pixel (u,v) to world (x,y)."""

    u, v = uv
    x = float(meta["left"]) + float(u) * float(meta["scale"])
    y = float(meta["top"]) - float(v) * float(meta["scale"])
    return np.array([x, y], dtype=np.float32)


def read_png_size(path: Path) -> tuple[int, int]:
    """Return image dimensions from the PNG header."""

    with path.open("rb") as fh:
        header = fh.read(8)
        if header != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a valid PNG file")
        length = int.from_bytes(fh.read(4), "big")
        chunk_type = fh.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"{path} missing IHDR chunk")
        width = int.from_bytes(fh.read(4), "big")
        height = int.from_bytes(fh.read(4), "big")
        _ = fh.read(length - 8)  # skip remaining IHDR payload
    return width, height


def load_occupancy_metadata(dataset_dir: Path) -> dict:
    """Reuse occupancy.json to infer world extents and z-range."""

    occ_json = dataset_dir / "occupancy.json"
    if not occ_json.is_file():
        raise FileNotFoundError(f"Missing occupancy.json in {dataset_dir}")

    with occ_json.open("r", encoding="utf-8") as fh:
        occ = json.load(fh)

    scale = float(occ.get("scale", 1.0))
    min_x, min_y, min_z = map(float, occ.get("min", (0.0, 0.0, 0.0)))
    max_x, max_y, max_z = map(float, occ.get("max", (0.0, 0.0, 0.0)))

    lower = occ.get("lower") or [min_x, min_y, min_z]
    upper = occ.get("upper") or [max_x, max_y, max_z]
    lower_z = float(lower[2])
    upper_z = float(upper[2])

    occ_png = dataset_dir / "occupancy.png"
    if not occ_png.is_file():
        raise FileNotFoundError(f"Missing occupancy.png in {dataset_dir}")

    width_px, height_px = read_png_size(occ_png)

    left = min_x
    right = left + width_px * scale
    top = max_y
    bottom = top - height_px * scale

    return {
        "width": int(width_px),
        "height": int(height_px),
        "scale": scale,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "lower_z": lower_z,
        "upper_z": upper_z,
    }


def load_raster_world_points_only(json_path: Path, *, swap_xy: bool = False) -> list[np.ndarray]:
    """Extract raster_world points as numpy arrays."""

    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    path_payload = payload.get("path", {})
    raster_world = path_payload.get("raster_world")
    if not raster_world:
        raise ValueError(f"Missing raster_world in {json_path}")

    points: list[np.ndarray] = []
    for idx, entry in enumerate(raster_world):
        try:
            x = float(entry["x"])
            y = float(entry["y"])
            z = float(entry.get("z", 0.0))
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Invalid raster_world entry #{idx} in {json_path}") from exc
        if swap_xy:
            points.append(np.array([y, x, z], dtype=np.float32))
        else:
            points.append(np.array([x, y, z], dtype=np.float32))
    return points


def load_raster_world_points(
    json_path: Path,
    *,
    swap_xy: bool = False,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Extract raster_world points and raster_pixel pairs."""

    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    path_payload = payload.get("path", {})
    if not isinstance(path_payload, dict):
        raise ValueError(f"Expected 'path' object in {json_path}, got {type(path_payload).__name__}")
    raster_world = path_payload.get("raster_world")
    raster_pixel = path_payload.get("raster_pixel")
    if not raster_world or not raster_pixel:
        raise ValueError(f"Missing raster_world or raster_pixel in {json_path}")
    if len(raster_world) != len(raster_pixel):
        raise ValueError(f"Length mismatch between raster_world and raster_pixel in {json_path}")

    points: list[np.ndarray] = []
    pixels: list[tuple[int, int]] = []
    for idx, (entry, pix) in enumerate(zip(raster_world, raster_pixel)):
        try:
            x = float(entry["x"])
            y = float(entry["y"])
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Invalid raster_world entry #{idx} in {json_path}") from exc
        if swap_xy:
            points.append(np.array([y, x], dtype=np.float32))
        else:
            points.append(np.array([x, y], dtype=np.float32))
        pixels.append((int(pix[0]), int(pix[1])))
    return points, pixels


def deduplicate_points(points: Sequence[np.ndarray], eps: float = 1e-4) -> list[np.ndarray]:
    """Remove consecutive duplicates within a tolerance."""

    if not points:
        return []
    deduped = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - deduped[-1]) > eps:
            deduped.append(point)
    return deduped


def sample_points(points: Sequence[np.ndarray], stride: int, eps: float = 1e-4) -> list[np.ndarray]:
    """Subsample points while guaranteeing the final point is kept."""

    if stride <= 1 or len(points) <= 2:
        sampled = list(points)
    else:
        sampled = [points[idx] for idx in range(0, len(points), stride)]
    if points and sampled:
        if np.linalg.norm(sampled[-1] - points[-1]) > eps:
            sampled.append(points[-1])
    return sampled


def derive_affine_transform(
    points: Sequence[np.ndarray],
    pixels: Sequence[tuple[int, int]],
    meta: dict,
) -> tuple[float, float, float, float]:
    """Solve for an affine transform mapping nav coordinates to scene coordinates."""

    n = len(points)
    if n < 2 or n != len(pixels):
        return 1.0, 0.0, 1.0, 0.0

    scale = float(meta["scale"])
    left = float(meta["left"])
    top = float(meta["top"])

    sum_x = sum(pt[0] for pt in points)
    sum_y = sum(pt[1] for pt in points)
    sum_x2 = sum(pt[0] * pt[0] for pt in points)
    sum_y2 = sum(pt[1] * pt[1] for pt in points)
    sum_map_x = 0.0
    sum_map_y = 0.0
    sum_x_map_x = 0.0
    sum_y_map_y = 0.0

    for pt, pix in zip(points, pixels):
        map_x = left + int(pix[0]) * scale
        map_y = top - int(pix[1]) * scale
        sum_map_x += map_x
        sum_map_y += map_y
        sum_x_map_x += pt[0] * map_x
        sum_y_map_y += pt[1] * map_y

    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y
    if abs(denom_x) < 1e-8 or abs(denom_y) < 1e-8:
        return 1.0, 0.0, 1.0, 0.0

    a_x = (n * sum_x_map_x - sum_x * sum_map_x) / denom_x
    b_x = (sum_map_x - a_x * sum_x) / n
    a_y = (n * sum_y_map_y - sum_y * sum_map_y) / denom_y
    b_y = (sum_map_y - a_y * sum_y) / n
    return a_x, b_x, a_y, b_y
