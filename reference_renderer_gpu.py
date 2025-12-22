#!/usr/bin/env python3
# search for "test" for comments to be removed after testing of the implementation is done

"""Render walkthrough frames for raster_world trajectories.

GPU-resident variant of the reference renderer. Both the base scene and actor
Gaussians are uploaded once and then reused across frames; per-frame PLY writes
and region crops are avoided.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import traceback
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

import imageio.v2 as imageio
import numpy as np
import torch
from plyfile import PlyElementParseError

from arguments import PipelineParams
from gaussian_renderer import render_or
from scene import GaussianModel
from scene.cameras import MiniCam

from utils import gaussian_ply_utils as ply_utils
from utils.graphics_utils import getProjectionMatrix


BASE_DIR = Path(__file__).resolve().parent
SCENES_DIR = BASE_DIR / "data" / "scenes"
# TASK_OUTPUT_DIR = BASE_DIR / "data" / "task_outputs_10w"
TASK_OUTPUT_DIR = BASE_DIR / "data" / "task_outputs_Jiankun_test"
# DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "path_video_frames"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "path_video_frames_Jiankun_test"
DEFAULT_ERROR_LOG = BASE_DIR / "jiankun_test_errors.log"
EPS = 1e-6
FORWARD_SMOOTH_BLEND = 0.35
DEFAULT_VIDEO_FPS = 10
DEFAULT_ACTOR_PATTERN = "*.ply"
DEFAULT_ACTOR_SPEED = 1.3
ACTOR_REGION_MARGIN = 6.0
STABILIZE_WINDOW = 5

FRAME_REGION_WINDOW = 4
FRAME_REGION_ACTOR_MARGIN = 1.5
FRAME_REGION_LOOKAHEAD_SCALE = 1.25
FRAME_REGION_MIN_MARGIN = 1.0
FRAME_REGION_MAX_POINTS = 180_000
FRAME_REGION_Z_WEIGHT = 0.5


def _format_bytes(num_bytes: int) -> str:
    """Human readable formatting for byte counts."""

    sign = "-" if num_bytes < 0 else ""
    value = float(abs(num_bytes))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{sign}{int(value)} {unit}"
            return f"{sign}{value:.2f} {unit}"
        value /= 1024.0
    return f"{sign}{value:.2f} TiB"


def _log_vram_usage(message: str, device: torch.device, before_bytes: int | None = None) -> int:
    """Print current and delta GPU memory usage when verbose mode is active."""

    current = torch.cuda.memory_allocated(device)
    if before_bytes is None:
        delta_str = ""
    else:
        delta = current - before_bytes
        delta_str = f" (Δ{_format_bytes(delta)})"
    print(f"[VERBOSE][VRAM] {message}: total={_format_bytes(current)}{delta_str}", flush=True)
    return current


@contextlib.contextmanager
def _cuda_oom_trace(label: str, device: torch.device, verbose: bool = False):
    """Context manager that augments CUDA OOM exceptions with additional diagnostics."""

    try:
        yield
    except RuntimeError as exc:
        message = str(exc)
        if "CUDA out of memory" not in message:
            raise

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_alloc = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        diagnostics = (
            f"[CUDA OOM @ {label}] "
            f"allocated={_format_bytes(allocated)} "
            f"reserved={_format_bytes(reserved)} "
            f"max_alloc={_format_bytes(max_alloc)} "
            f"max_reserved={_format_bytes(max_reserved)}"
        )
        if verbose:
            print(f"[VERBOSE][VRAM] {diagnostics}", flush=True)
        raise RuntimeError(diagnostics) from exc


@dataclass(frozen=True)
class ActorOptions:
    sequence_dir: Path
    pattern: str
    height: float
    follow_distance: float
    buffer_distance: float
    speed: float
    fps: float
    loop: bool
    foot_offset: float


@dataclass
class ActorSequenceFrame:
    base_data: np.ndarray


@dataclass
class ActorSequence:
    frames: list[ActorSequenceFrame]
    height: float
    hip_height: float
    columns: dict[str, int]
    dtype: np.dtype
    feature_rest_names: list[str]
    scale_names: list[str]
    rot_names: list[str]
    rest_dim: int
    max_sh_degree: int
    uniform_scale: bool


@dataclass
class ActorRenderFrame:
    xyz: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor


class RegionGaussianModel:
    """Subset of a GaussianModel restricted to specified indices."""

    def __init__(self, base: GaussianModel, indices: torch.Tensor):
        self.base = base
        self.indices = indices
        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree = base.max_sh_degree

        self._xyz = base.get_xyz.index_select(0, indices).detach().contiguous()
        self._features_dc = base.get_features_dc.index_select(0, indices).detach().contiguous()
        self._features_rest = base.get_features_rest.index_select(0, indices).detach().contiguous()
        self._opacity = base.get_opacity.index_select(0, indices).detach().contiguous()
        self._scaling = base.get_scaling.index_select(0, indices).detach().contiguous()
        self._rotation = base.get_rotation.index_select(0, indices).detach().contiguous()

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        return self._features_rest

    @property
    def get_features(self) -> torch.Tensor:
        if self._features_rest.shape[1] == 0:
            return self._features_dc
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation


@dataclass(frozen=True)
class ActorRuntime:
    options: ActorOptions
    sequence: ActorSequence


@dataclass
class ActorFramePlan:
    base_frame_index: int
    transform: np.ndarray
    actor_pos_xy: np.ndarray
    direction_xy: np.ndarray
    camera_offset: float


class PathSampler:
    """Helper to query positions and tangents along a 2D polyline."""

    def __init__(self, points: Sequence[np.ndarray]):
        if len(points) < 2:
            raise ValueError("PathSampler requires at least two points.")
        raw = np.asarray(points, dtype=np.float32)
        diffs = raw[1:] - raw[:-1]
        lengths = np.linalg.norm(diffs, axis=1)
        valid_mask = lengths > 1e-6
        if not np.any(valid_mask):
            raise ValueError("PathSampler received a zero-length polyline.")

        cleaned: list[np.ndarray] = [raw[0]]
        cleaned_vectors: list[np.ndarray] = []
        cleaned_lengths: list[float] = []
        for idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue
            cleaned.append(raw[idx + 1])
            cleaned_vectors.append(diffs[idx])
            cleaned_lengths.append(float(lengths[idx]))

        self.points = np.asarray(cleaned, dtype=np.float32)
        self.segment_vectors = np.asarray(cleaned_vectors, dtype=np.float32)
        self.segment_lengths = np.asarray(cleaned_lengths, dtype=np.float32)
        self.cumulative = np.concatenate(
            [np.array([0.0], dtype=np.float32), np.cumsum(self.segment_lengths)]
        )

    @property
    def total_length(self) -> float:
        return float(self.cumulative[-1])

    def position_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            direction = self.segment_vectors[0] / self.segment_lengths[0]
            return self.points[0] + direction * distance
        total = self.total_length
        if distance >= total:
            direction = self.segment_vectors[-1] / self.segment_lengths[-1]
            return self.points[-1] + direction * (distance - total)

        seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
        seg_offset = distance - self.cumulative[seg_idx]
        ratio = seg_offset / self.segment_lengths[seg_idx]
        return self.points[seg_idx] + self.segment_vectors[seg_idx] * ratio

    def direction_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            vec = self.segment_vectors[0]
        elif distance >= self.total_length:
            vec = self.segment_vectors[-1]
        else:
            seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
            vec = self.segment_vectors[seg_idx]
        norm = np.linalg.norm(vec)
        if norm < EPS:
            return np.array([0.0, 1.0], dtype=np.float32)
        return vec / norm


_DIGIT_PATTERN = re.compile(r"(\d+)")


# BEV helpers (unchanged)

def _world_to_pixel(meta: dict, xy: np.ndarray) -> tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    u = int(round((x - float(meta["left"])) / float(meta["scale"])))
    v = int(round((float(meta["top"]) - y) / float(meta["scale"])))
    return u, v


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        return img[..., :3]
    return img


def _draw_disk(img: np.ndarray, uv: tuple[int, int], r: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    u0, v0 = uv
    umin = max(0, u0 - r)
    umax = min(w - 1, u0 + r)
    vmin = max(0, v0 - r)
    vmax = min(h - 1, v0 + r)
    rr = r * r
    for v in range(vmin, vmax + 1):
        dv = v - v0
        for u in range(umin, umax + 1):
            du = u - u0
            if du * du + dv * dv <= rr:
                img[v, u, :] = color


def _draw_polyline(img: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 1, dotted: bool = False, dot_gap: int = 6) -> None:
    if len(pts) < 2:
        return
    h, w = img.shape[:2]

    def put(u: int, v: int) -> None:
        if 0 <= v < h and 0 <= u < w:
            img[v, u, :] = color
            if thickness > 1:
                for dv in range(-thickness + 1, thickness):
                    for du in range(-thickness + 1, thickness):
                        vv, uu = v + dv, u + du
                        if 0 <= vv < h and 0 <= uu < w:
                            img[vv, uu, :] = color

    for seg_idx, ((u0, v0), (u1, v1)) in enumerate(zip(pts[:-1], pts[1:])):
        du = u1 - u0
        dv = v1 - v0
        steps = int(max(abs(du), abs(dv))) + 1
        if steps <= 1:
            if (not dotted) or (seg_idx % dot_gap == 0):
                put(u0, v0)
            continue
        for t in range(steps + 1):
            if dotted and ((t // dot_gap) % 2 == 1):
                continue
            u = int(round(u0 + du * (t / steps)))
            v = int(round(v0 + dv * (t / steps)))
            put(u, v)


def _draw_text_lines(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 8)) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        x, y = origin
        for line in lines:
            draw.text((x, y), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            y += 14
        img[:] = np.array(pil_img)
    except Exception:
        x, y = origin
        for _ in lines:
            for dx in range(40):
                if y < img.shape[0] and x + dx < img.shape[1]:
                    img[y, x + dx, :] = (200, 200, 200)
            y += 6


def save_bev_debug_image(
    *,
    scene_id: str,
    label_id: str,
    meta: dict,
    camera_xy_seq: list[np.ndarray] | None,
    actor_xy_seq: list[np.ndarray] | None,
    out_path: Path,
    look_ahead: float,
    follow_points: int | None,
    total_length_m: float | None,
    fps: int = DEFAULT_VIDEO_FPS,
    actor_path_dotted: bool = False,
    mirror_bev_x: bool = True,
    mirror_bev_y: bool = True,
) -> None:
    def _mirror_pts(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not pts:
            return pts
        out = []
        for (u, v) in pts:
            uu = (w - 1 - u) if mirror_bev_x else u
            vv = (h - 1 - v) if mirror_bev_y else v
            out.append((uu, vv))
        return out

    occ_path = SCENES_DIR / scene_id / "occupancy.png"
    if not occ_path.is_file():
        w = int((meta["right"] - meta["left"]) / meta["scale"])
        h = int((meta["top"] - meta["bottom"]) / meta["scale"])
        base = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        base = imageio.imread(occ_path)
        base = _ensure_rgb(base.copy())
    h, w = base.shape[:2]

    cam_pts: list[tuple[int, int]] = []
    act_pts: list[tuple[int, int]] = []
    if camera_xy_seq:
        for xy in camera_xy_seq:
            cam_pts.append(_world_to_pixel(meta, xy))
    if actor_xy_seq:
        for xy in actor_xy_seq:
            act_pts.append(_world_to_pixel(meta, xy))
    cam_pts = _mirror_pts(cam_pts)
    act_pts = _mirror_pts(act_pts)
    if cam_pts:
        _draw_polyline(base, cam_pts, (255, 0, 255), thickness=2, dotted=False)
        _draw_disk(base, cam_pts[0], 4, (255, 255, 0))
        _draw_disk(base, cam_pts[-1], 4, (255, 0, 0))
    if act_pts:
        _draw_polyline(base, act_pts, (0, 255, 0), thickness=2, dotted=actor_path_dotted)
        _draw_disk(base, act_pts[0], 4, (0, 255, 255))
        _draw_disk(base, act_pts[-1], 4, (0, 128, 0))

    lines = [
        f"Scene: {scene_id}",
        f"Label: {label_id}",
        f"Image: {w}x{h}px  scale:{meta['scale']:.3f} m/px",
        f"Frames(camera): {len(cam_pts) if cam_pts else 0}",
        f"Frames(actor):  {len(act_pts) if act_pts else 0}",
        f"Follow distance (points): {follow_points if follow_points is not None else '-'}",
        f"Look-ahead (m): {look_ahead}",
        f"FPS: {fps}",
    ]
    if total_length_m is not None:
        lines.append(f"Path length: {total_length_m:.2f} m")

    _draw_text_lines(base, lines, origin=(8, 8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, base)


def natural_sort_key(path: Path) -> list[object]:
    parts = _DIGIT_PATTERN.split(path.stem)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    key.append(path.suffix.lower())
    return key


def list_actor_frame_paths(options: ActorOptions) -> list[Path]:
    if not options.sequence_dir.is_dir():
        return []

    pattern = options.pattern or "*.ply"
    initial = [path for path in options.sequence_dir.glob(pattern) if path.is_file()]
    initial = [path for path in initial if path.suffix.lower() == ".ply"]

    if not initial:
        initial = [path for path in options.sequence_dir.glob("*.ply") if path.is_file()]

    return sorted(initial, key=natural_sort_key)


def load_gaussian_ply(path: Path) -> ply_utils.GaussianPly:
    try:
        return ply_utils.GaussianPly.read(path)
    except PlyElementParseError as exc:
        raise ValueError(f"Unable to parse actor PLY: {path}") from exc


def rotation_matrix_z_np(theta: float) -> np.ndarray:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    if translation.shape != (3,):
        raise ValueError("translation must be length-3 vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


ACTOR_AXIS_ALIGNMENT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
HIP_HEIGHT_RATIO = 0.6


def load_actor_sequence(
    options: ActorOptions,
    *,
    debug: bool = False,
) -> ActorSequence:
    if not options.sequence_dir.is_dir():
        raise FileNotFoundError(f"Actor sequence directory not found: {options.sequence_dir}")

    ply_files = list_actor_frame_paths(options)
    if not ply_files:
        raise FileNotFoundError(
            f"No actor frame PLY files found in {options.sequence_dir}"
        )

    alignment_transform = np.eye(4, dtype=np.float64)
    alignment_transform[:3, :3] = ACTOR_AXIS_ALIGNMENT_MATRIX

    actor_plys: list[ply_utils.GaussianPly] = []
    z_values: list[np.ndarray] = []

    for ply_path in ply_files:
        ply = load_gaussian_ply(Path(ply_path))
        ply_utils.apply_transform_inplace(
            ply,
            alignment_transform,
            rotate_normals=True,
            rotate_sh=True,
        )
        actor_plys.append(ply)
        z_values.append(ply.data["z"].astype(np.float64))

    combined_z = np.concatenate(z_values)
    raw_min_z = float(np.min(combined_z))
    raw_max_z = float(np.max(combined_z))
    measured_height = max(raw_max_z - raw_min_z, EPS)
    target_height = options.height if options.height > 0.0 else measured_height
    scale_factor = target_height / measured_height

    if debug:
        print(
            f"[DEBUG] Actor sequence: {len(actor_plys)} frames, "
            f"raw height {measured_height:.3f} m, applying scale {scale_factor:.3f}",
            flush=True,
        )

    if not math.isclose(scale_factor, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        scale_transform = np.eye(4, dtype=np.float64)
        scale_transform[:3, :3] *= scale_factor
        for ply in actor_plys:
            ply_utils.apply_transform_inplace(
                ply,
                scale_transform,
                rotate_normals=True,
                rotate_sh=True,
            )

    global_min_z = min(float(np.min(ply.data["z"])) for ply in actor_plys)
    translate_transform = np.eye(4, dtype=np.float64)
    translate_transform[2, 3] = -global_min_z
    for ply in actor_plys:
        ply_utils.apply_transform_inplace(
            ply,
            translate_transform,
            rotate_normals=False,
            rotate_sh=False,
        )

    global_max_z = max(float(np.max(ply.data["z"])) for ply in actor_plys)
    adjusted_height = max(global_max_z, EPS)
    hip_height = adjusted_height * HIP_HEIGHT_RATIO

    first_ply = actor_plys[0]
    dtype_names = list(first_ply.data.dtype.names or ())
    feature_rest_names = sorted(
        [name for name in dtype_names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rest_dim = len(feature_rest_names) // 3 if feature_rest_names else 0
    scale_names = sorted(
        [name for name in dtype_names if name.startswith("scale")],
        key=lambda name: int(name.split("_")[-1]) if "_" in name else 0,
    )
    rot_names = sorted(
        [name for name in dtype_names if name.startswith("rot_")],
        key=lambda name: int(name.split("_")[-1]),
    )

    if rest_dim * 3 != len(feature_rest_names):
        raise ValueError("Unexpected spherical harmonic coefficient layout in actor PLY.")

    if rot_names and len(rot_names) != 4:
        raise ValueError("Actor PLY must provide quaternion components rot_0..rot_3.")
    if scale_names and len(scale_names) not in (1, 3):
        raise ValueError("Actor PLY scales must appear as scale_0/1/2 or a single scale column.")

    max_sh_degree = int(round(math.sqrt(rest_dim + 1) - 1)) if rest_dim > 0 else 0

    frames = [
        ActorSequenceFrame(base_data=np.array(ply.data, copy=True)) for ply in actor_plys
    ]

    return ActorSequence(
        frames=frames,
        height=float(adjusted_height),
        hip_height=float(hip_height),
        columns=dict(first_ply.columns),
        dtype=first_ply.data.dtype,
        feature_rest_names=feature_rest_names,
        scale_names=scale_names,
        rot_names=rot_names,
        rest_dim=rest_dim,
        max_sh_degree=max_sh_degree,
        uniform_scale=len(scale_names) == 1,
    )


def apply_transform_to_frame(
    base_frame: ActorSequenceFrame,
    sequence: ActorSequence,
    transform: np.ndarray,
) -> np.ndarray:
    data = np.array(base_frame.base_data, copy=True)
    ply = ply_utils.GaussianPly(
        ply=None,
        vertex=None,
        data=data,
        columns=sequence.columns,
    )
    ply_utils.apply_transform_inplace(
        ply,
        transform,
        rotate_normals=True,
        rotate_sh=True,
    )
    return ply.data


def actor_data_to_tensors(
    data: np.ndarray,
    sequence: ActorSequence,
    device: torch.device,
    *,
    verbose: bool = False,
    log_prefix: str | None = None,
) -> ActorRenderFrame:
    allocation_before = torch.cuda.memory_allocated(device) if verbose else None

    xyz_np = np.stack((data["x"], data["y"], data["z"]), axis=1).astype(np.float32)
    xyz = torch.from_numpy(xyz_np).to(device)

    dc_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
    if not all(name in data.dtype.names for name in dc_names):
        missing = [name for name in dc_names if name not in data.dtype.names]
        raise KeyError(f"Actor PLY missing DC SH coefficients: {missing}")
    features_dc_np = np.stack([data[name] for name in dc_names], axis=1).astype(np.float32)
    features_dc = torch.from_numpy(features_dc_np[:, :, None]).to(device).transpose(1, 2).contiguous()

    if sequence.rest_dim > 0:
        rest_np = np.stack(
            [data[name] for name in sequence.feature_rest_names],
            axis=1,
        ).astype(np.float32)
        rest_np = rest_np.reshape(data.shape[0], 3, sequence.rest_dim)
        features_rest = torch.from_numpy(rest_np.transpose(0, 2, 1)).to(device)
    else:
        features_rest = torch.zeros(
            (data.shape[0], 0, 3),
            dtype=torch.float32,
            device=device,
        )

    opacity_np = np.asarray(data["opacity"], dtype=np.float32).reshape(-1, 1)
    opacity = torch.from_numpy(opacity_np).to(device)

    if sequence.scale_names:
        if sequence.uniform_scale:
            scale_values = np.asarray(data[sequence.scale_names[0]], dtype=np.float32).reshape(-1, 1)
            scales_np = np.repeat(scale_values, 3, axis=1)
        else:
            scales_np = np.stack(
                [data[name] for name in sequence.scale_names],
                axis=1,
            ).astype(np.float32)
    else:
        scales_np = np.zeros((data.shape[0], 0), dtype=np.float32)
    scaling = torch.from_numpy(scales_np).to(device)

    rotation_np = np.stack(
        [data[name] for name in sequence.rot_names],
        axis=1,
    ).astype(np.float32)
    rotation = torch.from_numpy(rotation_np).to(device)

    if verbose:
        frame_points = xyz.shape[0]
        prefix = log_prefix or "Actor frame tensors"
        _log_vram_usage(
            f"{prefix} ({frame_points} gaussians)",
            device,
            allocation_before,
        )

    return ActorRenderFrame(
        xyz=xyz.contiguous(),
        features_dc=features_dc.contiguous(),
        features_rest=features_rest.contiguous(),
        opacity=opacity.contiguous(),
        scaling=scaling.contiguous(),
        rotation=rotation.contiguous(),
    )


class CombinedGaussianModel:
    """Reusable container that stores base scene gaussians plus an actor slice."""

    def __init__(self, base: GaussianModel, actor_frame: ActorRenderFrame):
        device = base.get_xyz.device
        dtype = base.get_xyz.dtype
        self.base_size = base.get_xyz.shape[0]
        actor_size = actor_frame.xyz.shape[0]

        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree = base.max_sh_degree

        self._xyz = torch.empty((self.base_size + actor_size, 3), device=device, dtype=dtype)
        self._xyz[: self.base_size] = base.get_xyz.detach()
        self._xyz[self.base_size :] = actor_frame.xyz

        dc_base = base.get_features_dc.detach()
        self._features_dc = torch.empty(
            (self.base_size + actor_size, dc_base.shape[1], dc_base.shape[2]),
            device=device,
            dtype=dc_base.dtype,
        )
        self._features_dc[: self.base_size] = dc_base
        self._features_dc[self.base_size :] = actor_frame.features_dc

        rest_base = base.get_features_rest.detach()
        if rest_base.shape[1] > 0:
            self._features_rest = torch.empty(
                (self.base_size + actor_size, rest_base.shape[1], rest_base.shape[2]),
                device=device,
                dtype=rest_base.dtype,
            )
            self._features_rest[: self.base_size] = rest_base
            self._features_rest[self.base_size :] = actor_frame.features_rest
        else:
            self._features_rest = torch.zeros(
                (self.base_size + actor_size, 0, 0),
                device=device,
                dtype=rest_base.dtype,
            )

        opacity_base = base.get_opacity.detach()
        self._opacity = torch.empty((self.base_size + actor_size, 1), device=device, dtype=opacity_base.dtype)
        self._opacity[: self.base_size] = opacity_base
        self._opacity[self.base_size :] = actor_frame.opacity

        scaling_base = base.get_scaling.detach()
        self._scaling = torch.empty((self.base_size + actor_size, scaling_base.shape[1]), device=device, dtype=scaling_base.dtype)
        self._scaling[: self.base_size] = scaling_base
        if actor_frame.scaling.shape[1] == 0:
            self._scaling[self.base_size :] = 0.0
        else:
            self._scaling[self.base_size :] = actor_frame.scaling

        rotation_base = base.get_rotation.detach()
        self._rotation = torch.empty((self.base_size + actor_size, rotation_base.shape[1]), device=device, dtype=rotation_base.dtype)
        self._rotation[: self.base_size] = rotation_base
        self._rotation[self.base_size :] = actor_frame.rotation

    def update_actor(self, actor_frame: ActorRenderFrame) -> None:
        self._xyz[self.base_size :] = actor_frame.xyz
        self._features_dc[self.base_size :] = actor_frame.features_dc
        if self._features_rest.shape[1] > 0:
            self._features_rest[self.base_size :] = actor_frame.features_rest
        self._opacity[self.base_size :] = actor_frame.opacity
        if actor_frame.scaling.shape[1] == 0:
            self._scaling[self.base_size :] = 0.0
        else:
            self._scaling[self.base_size :] = actor_frame.scaling
        self._rotation[self.base_size :] = actor_frame.rotation

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        return self._features_rest

    @property
    def get_features(self) -> torch.Tensor:
        if self._features_rest.shape[1] == 0:
            return self._features_dc
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation


def render_actor_camera_only_sequence(
    *,
    scene_id: str,
    label_id: str,
    actor_runtime: ActorRuntime,
    path_xy: Sequence[np.ndarray],
    floor_z: float,
    camera_z: float,
    look_ahead: float,
    look_down: float,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    device: torch.device,
    bg_color: torch.Tensor,
    width: int,
    height: int,
    fov_deg: float,
    znear: float,
    zfar: float,
    writer,
    video: bool,
    frames_dir: Path,
    frame_prefix: str,
    debug: bool,
    stabilize: bool,
    verbose: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    options = actor_runtime.options
    follow_points = max(int(options.follow_distance), 0)

    sampler = PathSampler(path_xy)
    distances = list(sampler.cumulative)
    total_steps = len(distances)

    frame_plans: list[ActorFramePlan] = []
    camera_positions: list[np.ndarray] = []
    actor_xy_seq_would_be: list[np.ndarray] = []
    cached_direction = np.array([0.0, 1.0], dtype=np.float32)

    if verbose:
        print(
            f"[VERBOSE] Scene {scene_id} / {label_id}: camera-only per-point path with {total_steps} frames.",
            flush=True,
        )

    for i, dist in enumerate(distances):
        actor_i = min(i + follow_points, total_steps - 1)
        direction_xy = sampler.direction_at(distances[actor_i])
        if np.linalg.norm(direction_xy) < 1e-6:
            direction_xy = cached_direction
        else:
            cached_direction = direction_xy

        camera_xy = path_xy[i]
        actor_pos_xy = path_xy[actor_i]
        actor_xy_seq_would_be.append(np.array(actor_pos_xy, dtype=np.float32))

        frame_plans.append(
            ActorFramePlan(
                base_frame_index=0,
                transform=np.eye(4, dtype=np.float64),
                actor_pos_xy=np.array(camera_xy, dtype=np.float32),
                direction_xy=np.array(direction_xy, dtype=np.float32),
                camera_offset=0.0,
            )
        )
        camera_positions.append(np.array([camera_xy[0], camera_xy[1], camera_z], dtype=np.float32))

    if not frame_plans:
        return [], []

    direction_window = STABILIZE_WINDOW if stabilize else 1
    prev_forward: np.ndarray | None = None
    frame_counter = 0

    for idx, _ in enumerate(frame_plans):
        camera_position = camera_positions[idx]
        forward = forward_direction(camera_positions, idx, window=direction_window)
        if np.linalg.norm(forward[:2]) < EPS:
            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if stabilize and prev_forward is not None:
            blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + forward * FORWARD_SMOOTH_BLEND
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm > EPS:
                forward = (blended / blended_norm).astype(np.float32)
        prev_forward = forward.copy()

        target_xy = camera_position[:2] - forward[:2] * look_ahead
        target_z = camera_position[2] - abs(look_down)
        target_z = max(target_z, floor_z + 0.05)
        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

        camera = build_perspective_camera(
            position=camera_position,
            target=target,
            width=width,
            height=height,
            fov_deg=fov_deg,
            znear=znear,
            zfar=zfar,
            device=device,
        )

        with _cuda_oom_trace(
            f"Scene {scene_id} / {label_id}: camera-only render_or frame {idx}",
            device,
            verbose,
        ):
            img_pkg = render_or(camera, gaussians, pipeline, bg_color=bg_color, orthographic=False)
        render = img_pkg["render"].detach().cpu().numpy()
        render_uint8 = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
        render_uint8 = np.rot90(render_uint8, k=2)

        if video:
            writer.append_data(render_uint8)
        else:
            frame_path = frames_dir / f"{frame_prefix}_{frame_counter:04d}.png"
            imageio.imwrite(frame_path, render_uint8)
        frame_counter += 1

        if verbose and (idx % 10 == 0 or idx == total_steps - 1):
            print(
                f"[VERBOSE] Scene {scene_id} / {label_id}: camera-only frame {idx + 1}/{total_steps} complete.",
                flush=True,
            )

    camera_xy_seq = [pos[:2].copy() for pos in camera_positions]
    return camera_xy_seq, actor_xy_seq_would_be


def render_actor_follow_sequence(
    *,
    scene_id: str,
    label_id: str,
    actor_runtime: ActorRuntime,
    path_xy: Sequence[np.ndarray],
    floor_z: float,
    camera_z: float,
    look_ahead: float,
    look_down: float,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    device: torch.device,
    bg_color: torch.Tensor,
    width: int,
    height: int,
    fov_deg: float,
    znear: float,
    zfar: float,
    writer,
    video: bool,
    frames_dir: Path,
    frame_prefix: str,
    debug: bool,
    stabilize: bool,
    verbose: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    options = actor_runtime.options
    sequence = actor_runtime.sequence
    if not sequence.frames:
        raise ValueError("Actor sequence is empty; cannot render.")

    sampler = PathSampler(path_xy)
    distances = list(sampler.cumulative)
    total_steps = len(distances)

    anim_step = options.fps / float(DEFAULT_VIDEO_FPS)
    anim_cursor = 0.0
    num_actor_frames = len(sequence.frames)

    actor_ground_z = floor_z + options.foot_offset
    follow_points = max(int(options.follow_distance), 0)

    frame_plans: list[ActorFramePlan] = []
    camera_positions: list[np.ndarray] = []
    cached_direction = np.array([0.0, 1.0], dtype=np.float32)

    if verbose:
        print(
            f"[VERBOSE] Scene {scene_id} / {label_id}: rendering actor path with {total_steps} frames (per-point).",
            flush=True,
        )

    for i, dist in enumerate(distances):
        actor_i = min(i + follow_points, total_steps - 1)

        direction_xy = sampler.direction_at(distances[actor_i])
        if np.linalg.norm(direction_xy) < 1e-6:
            direction_xy = cached_direction
        else:
            cached_direction = direction_xy

        theta = math.atan2(direction_xy[0], direction_xy[1])
        rotation_np = rotation_matrix_z_np(theta)

        actor_pos_xy = path_xy[actor_i]
        translation_vec = np.array([actor_pos_xy[0], actor_pos_xy[1], actor_ground_z], dtype=np.float64)
        transform = build_transform_matrix(rotation_np, translation_vec)

        if options.loop:
            anim_idx = int(anim_cursor) % num_actor_frames
        else:
            anim_idx = min(int(anim_cursor), num_actor_frames - 1)
        anim_cursor += anim_step

        camera_xy = path_xy[i]
        frame_plans.append(
            ActorFramePlan(
                base_frame_index=anim_idx,
                transform=transform,
                actor_pos_xy=np.array(actor_pos_xy, dtype=np.float32),
                direction_xy=np.array(direction_xy, dtype=np.float32),
                camera_offset=0.0,
            )
        )
        camera_positions.append(np.array([camera_xy[0], camera_xy[1], camera_z], dtype=np.float32))

    if not frame_plans:
        return [], []

    frame_counter = 0
    prev_forward: np.ndarray | None = None
    direction_window = STABILIZE_WINDOW if stabilize else 1
    combined_model: CombinedGaussianModel | None = None

    for idx, plan in enumerate(frame_plans):
        base_frame = sequence.frames[plan.base_frame_index]
        actor_data = apply_transform_to_frame(base_frame, sequence, plan.transform)
        log_prefix = f"Scene {scene_id} / {label_id}: actor frame {idx}" if verbose else None
        with _cuda_oom_trace(
            f"Scene {scene_id} / {label_id}: actor frame tensor upload {idx}",
            device,
            verbose,
        ):
            actor_render = actor_data_to_tensors(
                actor_data,
                sequence,
                device,
                verbose=verbose,
                log_prefix=log_prefix,
            )

        if combined_model is None:
            combined_model = CombinedGaussianModel(gaussians, actor_render)
        else:
            combined_model.update_actor(actor_render)

        camera_position = camera_positions[idx]

        forward = forward_direction(camera_positions, idx, window=direction_window)
        if np.linalg.norm(forward[:2]) < EPS:
            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if stabilize and prev_forward is not None:
            blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + forward * FORWARD_SMOOTH_BLEND
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm > EPS:
                forward = (blended / blended_norm).astype(np.float32)
                forward[2] = 0.0
        prev_forward = forward.copy()

        target_xy = camera_position[:2] - forward[:2] * look_ahead
        target_z = camera_position[2] - look_down
        target_z = max(target_z, floor_z + 0.05)
        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

        camera = build_perspective_camera(
            position=camera_position,
            target=target,
            width=width,
            height=height,
            fov_deg=fov_deg,
            znear=znear,
            zfar=zfar,
            device=device,
        )

        with _cuda_oom_trace(
            f"Scene {scene_id} / {label_id}: render_or frame {idx}",
            device,
            verbose,
        ):
            img_pkg = render_or(camera, combined_model, pipeline, bg_color=bg_color, orthographic=False)
        render = img_pkg["render"].detach().cpu().numpy()
        render_uint8 = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
        render_uint8 = np.rot90(render_uint8, k=2)

        if video:
            writer.append_data(render_uint8)
        else:
            frame_path = frames_dir / f"{frame_prefix}_{frame_counter:04d}.png"
            imageio.imwrite(frame_path, render_uint8)
        frame_counter += 1

        if verbose and (idx % 10 == 0 or idx == total_steps - 1):
            print(
                f"[VERBOSE] Scene {scene_id} / {label_id}: frame {idx + 1}/{total_steps} complete.",
                flush=True,
            )

        del actor_render

    if verbose:
        print(
            f"[VERBOSE] Scene {scene_id} / {label_id}: actor rendering complete ({frame_counter} frames).",
            flush=True,
        )

    camera_xy_seq = [pos[:2].copy() for pos in camera_positions]
    actor_xy_seq = [plan.actor_pos_xy.copy() for plan in frame_plans]
    return camera_xy_seq, actor_xy_seq


def resolve_label_directory(scene_task_dir: Path) -> Path | None:
    label_paths_dir = scene_task_dir / "label_paths"
    if label_paths_dir.is_dir():
        return label_paths_dir
    if scene_task_dir.is_dir() and any(scene_task_dir.glob("*.json")):
        return scene_task_dir
    return None


class OrthoMiniCam:
    """Orthographic camera mirroring batch_verify's configuration."""

    def __init__(
        self,
        width: int,
        height: int,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
        half_width: float,
        half_height: float,
        znear: float,
        zfar: float,
    ) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVy = 1.0
        self.FoVx = 1.0
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self._half_width = half_width
        self._half_height = half_height
        view_inv = torch.inverse(world_view_transform)
        self.camera_center = view_inv[3][:3]

    def get_full_proj_transform(self, orthographic: bool = False):
        if not orthographic:
            return self.full_proj_transform
        return self._half_width, self._half_height, self.full_proj_transform


def read_png_size(path: Path) -> tuple[int, int]:
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
        _ = fh.read(length - 8)
    return width, height


def load_occupancy_metadata(dataset_dir: Path) -> dict:
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


def find_ply_file(dataset_dir: Path) -> Path:
    preferred_names = (
        "debug-decompressed.ply",
        "decompressed.ply",
        "3dgs_decompressed.ply",
    )
    for name in preferred_names:
        candidate = dataset_dir / name
        if candidate.is_file():
            return candidate

    for candidate in sorted(dataset_dir.glob("*.ply")):
        if candidate.suffix == ".ply":
            return candidate
    raise FileNotFoundError(f"No .ply file found in {dataset_dir}")


def load_raster_world_points(
    json_path: Path,
    *,
    swap_xy: bool = False,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    path_payload = payload.get("path", {})
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
    if not points:
        return []
    deduped = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - deduped[-1]) > eps:
            deduped.append(point)
    return deduped


def sample_points(points: Sequence[np.ndarray], stride: int, eps: float = 1e-4) -> list[np.ndarray]:
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


def get_forward_fn(args):
    return forward_direction_beta if getattr(args, "use_forward_beta", False) else forward_direction


def forward_direction(points: Sequence[np.ndarray], idx: int, window: int = 1) -> np.ndarray:
    if len(points) == 1:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    accum = np.zeros(2, dtype=np.float32)
    count = 0
    max_step = max(1, int(window))

    for step in range(1, max_step + 1):
        next_idx = min(idx + step, len(points) - 1)
        delta = points[next_idx][:2] - points[idx][:2]
        if np.linalg.norm(delta) > 1e-4:
            accum += delta
            count += 1

    for step in range(1, max_step + 1):
        prev_idx = max(idx - step, 0)
        delta = points[idx][:2] - points[prev_idx][:2]
        if np.linalg.norm(delta) > 1e-4:
            accum += delta
            count += 1

    if count == 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    direction_xy = accum / float(count)
    norm = np.linalg.norm(direction_xy)
    if norm < 1e-4:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([direction_xy[0] / norm, direction_xy[1] / norm, 0.0], dtype=np.float32)


def forward_direction_beta(points: Sequence[np.ndarray], idx: int, window: int = 1) -> np.ndarray:
    n = len(points)
    if n == 1:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    w = max(1, int(window))
    accum = np.zeros(2, dtype=np.float64)
    weight_sum = 0.0

    for k in range(1, w + 1):
        i0 = max(idx - k, 0)
        i1 = min(idx + k, n - 1)
        if i1 > idx:
            d_fwd = points[i1][:2] - points[idx][:2]
            nf = np.linalg.norm(d_fwd)
            if nf > 1e-6:
                wf = k
                accum += wf * (d_fwd / nf)
                weight_sum += wf
        if i0 < idx:
            d_bwd = points[idx][:2] - points[i0][:2]
            nb = np.linalg.norm(d_bwd)
            if nb > 1e-6:
                wb = k
                accum += wb * (d_bwd / nb)
                weight_sum += wb

    if weight_sum < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    v = accum / weight_sum
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([v[0] / nv, v[1] / nv, 0.0], dtype=np.float32)


def build_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = -eye + target
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


def build_orthographic_camera(
    position: np.ndarray,
    meta: dict,
    device: torch.device,
    width: int,
    height: int,
) -> OrthoMiniCam:
    half_width = 0.5 * (meta["right"] - meta["left"])
    half_height = 0.5 * (meta["top"] - meta["bottom"])
    znear = 0.01
    zfar = (position[2] - meta["lower_z"]) + 1.0

    world_view_np = np.array(
        [
            [-1.0, 0.0, 0.0, position[0]],
            [0.0, -1.0, 0.0, position[1]],
            [0.0, 0.0, -1.0, position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    left_cam = -half_width
    right_cam = half_width
    top_cam = -half_height
    bottom_cam = half_height

    projection_np = np.array(
        [
            [2.0 / (right_cam - left_cam), 0.0, 0.0, -(right_cam + left_cam) / (right_cam - left_cam)],
            [0.0, 2.0 / (top_cam - bottom_cam), 0.0, -(top_cam + bottom_cam) / (top_cam - bottom_cam)],
            [0.0, 0.0, -2.0 / (zfar - znear), -(zfar + znear) / (zfar - znear)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    world_view_transform = torch.tensor(world_view_np, device=device).transpose(0, 1)
    projection_matrix = torch.tensor(projection_np, device=device).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)).squeeze(0)

    return OrthoMiniCam(
        width=width,
        height=height,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        half_width=half_width,
        half_height=half_height,
        znear=znear,
        zfar=zfar,
    )


def render_path_frames(
    scene_id: str,
    json_path: Path,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    device: torch.device,
    bg_color: torch.Tensor,
    meta: dict,
    output_dir: Path,
    stride: int,
    height_offset: float,
    look_ahead: float,
    look_down: float,
    width: int,
    height: int,
    fov_deg: float,
    znear: float,
    zfar: float,
    overwrite: bool,
    view_mode: str,
    swap_xy: bool,
    stabilize: bool,
    video: bool,
    debug: bool,
    mirror_translation: bool,
    render_bev: bool,
    actor_runtime: ActorRuntime | None,
    verbose: bool,
    hide_actor_enabled: bool = False,
    mirror_bev_x: bool = True,
    mirror_bev_y: bool = True,
) -> None:
    raw_points, raster_pixels = load_raster_world_points(json_path, swap_xy=swap_xy)
    a_x, b_x, a_y, b_y = derive_affine_transform(raw_points, raster_pixels, meta)
    transformed = [
        np.array([a_x * pt[0] + b_x, a_y * pt[1] + b_y], dtype=np.float32)
        for pt in raw_points
    ]

    points_xy = deduplicate_points(transformed)
    if len(points_xy) < 2:
        raise ValueError(f"Need at least two distinct points in {json_path}")
    sampled_xy = sample_points(points_xy, stride)
    if len(sampled_xy) < 2:
        sampled_xy = points_xy

    floor_z = meta["lower_z"]
    ceiling = meta["upper_z"]
    camera_z = ceiling + height_offset

    if mirror_translation:
        center_x = 0.5 * (meta["left"] + meta["right"])
        center_y = 0.5 * (meta["top"] + meta["bottom"])
        path_xy = [
            np.array([center_x * 2.0 - pt[0], center_y * 2.0 - pt[1]], dtype=np.float32)
            for pt in sampled_xy
        ]
    else:
        path_xy = [np.array([pt[0], pt[1]], dtype=np.float32) for pt in sampled_xy]

    positions = [
        np.array([xy[0], xy[1], camera_z], dtype=np.float32) for xy in path_xy
    ]

    if debug:
        print(f"[DEBUG] Processing label config: {json_path}", flush=True)
        print(
            f"[DEBUG] Affine transform: x' = {a_x:.6f} * x + {b_x:.6f}, "
            f"y' = {a_y:.6f} * y + {b_y:.6f} (swap_xy={swap_xy})",
            flush=True,
        )
        raw_preview = [tuple(map(float, pt)) for pt in raw_points[:5]]
        preview = [tuple(map(float, pts)) for pts in sampled_xy[:5]]
        print(
            f"[DEBUG] Raw points: {raw_preview} -> transformed: {preview}",
            flush=True,
        )
        if mirror_translation:
            mirrored_preview = [
                (
                    float(2.0 * center_x - pt[0]),
                    float(2.0 * center_y - pt[1]),
                )
                for pt in sampled_xy[:5]
            ]
            print(
                f"[DEBUG] Mirrored (translation-adjusted) XY: {mirrored_preview}",
                flush=True,
            )

    frames_dir = output_dir / scene_id / json_path.stem
    video_dir = output_dir / scene_id
    if video:
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{json_path.stem}.mp4"
        if not overwrite and video_path.exists():
            print(f"  Skipping {json_path.stem}: 视频已存在。")
            return
    else:
        frames_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            existing = list(frames_dir.glob("frame_*.png"))
            if existing:
                print(f"  Skipping {json_path.stem}: frames already exist ({len(existing)} files).")
                return

    with torch.no_grad():
        prev_forward: np.ndarray | None = None
        direction_window = STABILIZE_WINDOW if stabilize else 1
        writer_ctx = (
            imageio.get_writer(
                video_path,
                mode="I",
                fps=DEFAULT_VIDEO_FPS,
            )
            if video
            else contextlib.nullcontext()
        )

        with writer_ctx as writer:
            if verbose:
                print(
                    f"[VERBOSE] Stabilize={'on' if stabilize else 'off'}, "
                    f"window={direction_window}, "
                    f"blend={FORWARD_SMOOTH_BLEND:.2f}",
                    flush=True,
                )
            if actor_runtime is not None:
                if view_mode != "forward":
                    raise ValueError("Animated actor rendering currently supports --view-mode forward only.")
                if hide_actor_enabled:
                    if verbose:
                        print(
                            f"[VERBOSE] Scene {scene_id} / {json_path.stem}: hiding actor during render.",
                            flush=True,
                        )
                    render_result = render_actor_camera_only_sequence(
                        scene_id=scene_id,
                        label_id=json_path.stem,
                        actor_runtime=actor_runtime,
                        path_xy=path_xy,
                        floor_z=floor_z,
                        camera_z=camera_z,
                        look_ahead=look_ahead,
                        look_down=look_down,
                        gaussians=gaussians,
                        pipeline=pipeline,
                        device=device,
                        bg_color=bg_color,
                        width=width,
                        height=height,
                        fov_deg=fov_deg,
                        znear=znear,
                        zfar=zfar,
                        writer=writer,
                        video=video,
                        frames_dir=frames_dir,
                        frame_prefix="frame",
                        debug=debug,
                        stabilize=stabilize,
                        verbose=verbose,
                    )
                else:
                    render_result = render_actor_follow_sequence(
                        scene_id=scene_id,
                        label_id=json_path.stem,
                        actor_runtime=actor_runtime,
                        path_xy=path_xy,
                        floor_z=floor_z,
                        camera_z=camera_z,
                        look_ahead=look_ahead,
                        look_down=look_down,
                        gaussians=gaussians,
                        pipeline=pipeline,
                        device=device,
                        bg_color=bg_color,
                        width=width,
                        height=height,
                        fov_deg=fov_deg,
                        znear=znear,
                        zfar=zfar,
                        writer=writer,
                        video=video,
                        frames_dir=frames_dir,
                        frame_prefix="frame",
                        debug=debug,
                        stabilize=stabilize,
                        verbose=verbose,
                    )
                if render_bev:
                    if verbose:
                        print(
                            f"[VERBOSE] Scene {scene_id} / {json_path.stem}: saving BEV debug image.",
                            flush=True,
                        )
                    bev_dir = output_dir / scene_id
                    bev_path = bev_dir / f"{json_path.stem}_BEV.png"

                    if actor_runtime is not None:
                        if hide_actor_enabled:
                            cam_seq, act_seq = render_result
                            actor_path_dotted = True
                        else:
                            cam_seq, act_seq = render_result
                            actor_path_dotted = False
                    else:
                        cam_seq = [p[:2].copy() for p in positions]
                        act_seq = None
                        actor_path_dotted = False

                    sampler_for_len = PathSampler(path_xy)
                    total_len = sampler_for_len.total_length

                    save_bev_debug_image(
                        scene_id=scene_id,
                        label_id=json_path.stem,
                        meta=meta,
                        camera_xy_seq=cam_seq,
                        actor_xy_seq=act_seq,
                        out_path=bev_path,
                        look_ahead=look_ahead,
                        follow_points=(actor_runtime.options.follow_distance if actor_runtime is not None else None),
                        total_length_m=total_len,
                        fps=DEFAULT_VIDEO_FPS,
                        actor_path_dotted=actor_path_dotted,
                        mirror_bev_x=mirror_bev_x,
                        mirror_bev_y=mirror_bev_y,
                    )
            else:
                total_positions = len(positions)
                if verbose:
                    print(
                        f"[VERBOSE] Scene {scene_id} / {json_path.stem}: rendering {total_positions} frames.",
                        flush=True,
                    )

                for idx, position in enumerate(positions):
                    if view_mode == "forward":
                        forward = forward_direction(positions, idx, window=direction_window)
                        if np.linalg.norm(forward[:2]) < EPS:
                            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                        if stabilize and prev_forward is not None:
                            blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + forward * FORWARD_SMOOTH_BLEND
                            blended_norm = float(np.linalg.norm(blended))
                            if blended_norm > EPS:
                                forward = (blended / blended_norm).astype(np.float32)
                                forward[2] = 0.0
                        prev_forward = forward.copy()

                        target_xy = position[:2] + forward[:2] * look_ahead
                        target_z = position[2] - look_down
                        target_z = max(target_z, floor_z + 0.05)
                        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
                        camera = build_perspective_camera(
                            position=position,
                            target=target,
                            width=width,
                            height=height,
                            fov_deg=fov_deg,
                            znear=znear,
                            zfar=zfar,
                            device=device,
                        )
                        orthographic = False
                    elif view_mode == "topdown":
                        camera = build_orthographic_camera(
                            position=position,
                            meta=meta,
                            device=device,
                            width=width,
                            height=height,
                        )
                        orthographic = True
                    else:
                        raise ValueError(f"Unsupported view mode: {view_mode}")

                    with _cuda_oom_trace(
                        f"Scene {scene_id} / {json_path.stem}: base render_or frame {idx}",
                        device,
                        verbose,
                    ):
                        img_pkg = render_or(camera, gaussians, pipeline, bg_color=bg_color, orthographic=orthographic)
                    render = img_pkg["render"].detach().cpu().numpy()
                    render_uint8 = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    if orthographic:
                        render_uint8 = np.rot90(render_uint8, k=1)
                    else:
                        render_uint8 = np.rot90(render_uint8, k=2)

                    if video:
                        writer.append_data(render_uint8)
                    else:
                        frame_path = frames_dir / f"frame_{idx:04d}.png"
                        imageio.imwrite(frame_path, render_uint8)

                    if verbose and (idx % 10 == 0 or idx == total_positions - 1):
                        print(
                            f"[VERBOSE] Scene {scene_id} / {json_path.stem}: frame {idx + 1}/{total_positions} complete.",
                            flush=True,
                        )

            if verbose:
                print(
                    f"[VERBOSE] Scene {scene_id} / {json_path.stem}: render finished.",
                    flush=True,
                )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Render frames along raster_world navigation paths.")
    parser.add_argument(
        "--scene",
        nargs="+",
        default=None,
        help="Scene identifier(s) present in both data/scenes and data/task_outputs_10w.",
    )
    parser.add_argument(
        "--label-id",
        action="append",
        dest="label_ids",
        help="Optional label-path identifier(s) to restrict rendering (e.g. '59' or '59.json'). "
        "Repeat this flag to specify multiple IDs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every Nth point along the raster_world polyline (default: 1).",
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0,
        help="Add this many meters above occupancy upper_z when placing the camera (default: 1.0).",
    )
    parser.add_argument(
        "--view-mode",
        choices=("topdown", "forward"),
        default="forward",
        help="Camera orientation: 'topdown' matches batch_verify, 'forward' follows the path direction (default: topdown).",
    )
    parser.add_argument(
        "--look-ahead",
        type=float,
        default=2,
        help="Distance in meters to look ahead along the path direction (default: 0.5).",
    )
    parser.add_argument(
        "--look-down",
        type=float,
        default=0.1,
        help="Vertical offset in meters to tilt the camera downward (default: 0.1).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(960, 720),
        help="Output image resolution (default: 960 720).",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=70.0,
        help="Vertical field of view in degrees (default: 70).",
    )
    parser.add_argument(
        "--znear",
        type=float,
        default=0.001,
        help="Near clipping plane distance (default: 0.05).",
    )
    parser.add_argument(
        "--zfar",
        type=float,
        default=30.0,
        help="Far clipping plane distance (default: 30.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for rendered frames (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render frames even if target directory already contains images.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Limit the number of label-path JSON files processed per scene.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印调试信息，例如当前加载的 PLY 文件路径。",
    )
    parser.add_argument(
        "--swap-xy",
        action="store_true",
        help="在读取 raster_world 坐标时交换 x / y。",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        default=True,
        help="启用前向视角方向平滑以缓解相机抖动。",
    )
    parser.add_argument(
        "--video",
        action=BooleanOptionalAction,
        default=True,
        help="输出合成的 MP4 视频（默认开启）。使用 --no-video 可改为保存逐帧 PNG。",
    )
    parser.add_argument(
        "--no-mirror-translation",
        action="store_true",
        help="默认会将 XY 位置相对于场景中心做一次反射修正；使用该选项可关闭此操作。",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERROR_LOG,
        help=f"将渲染失败信息追加写入指定日志文件（默认：{DEFAULT_ERROR_LOG}）。",
    )
    parser.add_argument(
        "--verbose",
        action=BooleanOptionalAction,
        default=False,
        help="输出详细的渲染进度信息。",
    )
    parser.add_argument(
        "--hide-actor",
        action="store_true",
        default=False,
        help="禁用动画角色叠加渲染。"
    )
    parser.add_argument(
        "--use-forward-beta",
        action="store_true",
        help="Use the improved forward_direction_beta function for smoother turns."
    )

    parser.add_argument(
        "--actor-seq-dir",
        type=Path,
        default=None,
        help="Directory containing animated actor .ply frames to overlay during rendering.",
    )
    parser.add_argument(
        "--actor-pattern",
        type=str,
        default=DEFAULT_ACTOR_PATTERN,
        help=f"Glob pattern for actor frames (default: {DEFAULT_ACTOR_PATTERN}).",
    )
    parser.add_argument(
        "--actor-height",
        type=float,
        default=1.7,
        help="Target actor height in meters after normalisation (default: 1.7).",
    )
    parser.add_argument(
        "--actor-speed",
        type=float,
        default=DEFAULT_ACTOR_SPEED,
        help=f"Actor walking speed in m/s (default: {DEFAULT_ACTOR_SPEED}).",
    )
    parser.add_argument(
        "--actor-fps",
        type=float,
        default=DEFAULT_VIDEO_FPS,
        help=f"Playback rate for actor animation frames (default: {DEFAULT_VIDEO_FPS}).",
    )
    parser.add_argument(
        "--follow-distance",
        type=float,
        default=100,
        help="Preferred trailing camera distance behind the actor in meters (default: 1).",
    )
    parser.add_argument(
        "--follow-buffer",
        type=float,
        default=0.5,
        help="Camera distance at sequence start and end (default: 0.5 meters).",
    )
    parser.add_argument(
        "--actor-foot-offset",
        type=float,
        default=0.0,
        help="Additional offset applied to actor vertical placement (default: 0.0).",
    )
    parser.add_argument(
        "--actor-no-loop",
        dest="actor_loop",
        action="store_false",
        help="Disable looping of actor animation frames (default: loop).",
    )
    parser.add_argument(
        "--show-BEV",
        action=BooleanOptionalAction,
        default=False,
        help="Save a bird's-eye-view debug image of camera (magenta) and actor (green) paths next to the video."
    )
    parser.add_argument(
        "--bev-mirror-x",
        action=BooleanOptionalAction,
        default=True,
        help="Mirror BEV paths across the X axis (u -> W-1-u) before drawing (default: on)."
    )
    parser.add_argument(
        "--bev-mirror-y",
        action=BooleanOptionalAction,
        default=True,
        help="Mirror BEV paths across the Y axis (v -> H-1-v) before drawing (default: on)."
    )

    parser.set_defaults(actor_loop=True)
    return parser


def normalise_label_ids(raw_ids: list[str] | None) -> set[str]:
    if not raw_ids:
        return set()
    normalised = set()
    for entry in raw_ids:
        parts = entry.split(",")
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue
            stem = Path(cleaned).stem
            normalised.add(stem)
    return normalised


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for rendering but not available.")
    device = torch.device("cuda")

    label_filter = normalise_label_ids(args.label_ids)
    width, height = args.resolution

    pipeline_parser = ArgumentParser(description="Pipeline parameters placeholder")
    pipeline = PipelineParams(pipeline_parser)
    pipeline.antialiasing = True
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    debug_enabled = bool(args.debug)
    verbose_enabled = bool(args.verbose)
    swap_xy_enabled = bool(args.swap_xy)
    stabilize_enabled = bool(args.stabilize)
    hide_actor_enabled = bool(args.hide_actor)
    video_enabled = bool(args.video)
    actor_runtime: ActorRuntime | None = None

    if args.actor_seq_dir is not None:
        actor_options = ActorOptions(
            sequence_dir=args.actor_seq_dir,
            pattern=args.actor_pattern,
            height=float(args.actor_height),
            follow_distance=float(args.follow_distance),
            buffer_distance=float(args.follow_buffer),
            speed=float(args.actor_speed),
            fps=float(args.actor_fps),
            loop=bool(args.actor_loop),
            foot_offset=float(args.actor_foot_offset),
        )
        if actor_options.fps <= 0.0:
            raise ValueError("Actor FPS must be positive.")
        if actor_options.speed <= 0.0:
            raise ValueError("Actor walking speed must be positive.")
        if actor_options.buffer_distance > actor_options.follow_distance:
            raise ValueError("Follow buffer must be less than or equal to follow distance.")
        actor_sequence = load_actor_sequence(
            actor_options,
            debug=debug_enabled,
        )
        actor_runtime = ActorRuntime(options=actor_options, sequence=actor_sequence)
        if debug_enabled:
            print(
                f"[DEBUG] Actor sequence loaded from {args.actor_seq_dir} "
                f"({len(actor_sequence.frames)} frames, height {actor_sequence.height:.3f} m).",
                flush=True,
            )
        elif verbose_enabled:
            print(
                f"[VERBOSE] Actor sequence ready: {len(actor_sequence.frames)} frames from {args.actor_seq_dir}.",
                flush=True,
            )

    if args.scene:
        scene_ids = list(dict.fromkeys(args.scene))
    else:
        if not TASK_OUTPUT_DIR.is_dir():
            raise FileNotFoundError(f"Task output directory not found: {TASK_OUTPUT_DIR}")
        scene_ids = sorted(
            entry.name
            for entry in TASK_OUTPUT_DIR.iterdir()
            if entry.is_dir() and resolve_label_directory(entry) is not None
        )
        scene_ids = scene_ids[:1]  # test
        if not scene_ids:
            print(f"⚠️  在 {TASK_OUTPUT_DIR} 下未找到任何场景，程序结束。")
            return
        print(f"未提供 --scene，默认遍历 {len(scene_ids)} 个包含标签数据的场景。", flush=True)

    total_scenes = len(scene_ids)
    error_log_path: Path = args.error_log
    error_log_file: TextIO | None = None
    error_count = 0
    processed_scenes = 0
    overall_labels = 0

    def ensure_log_file() -> TextIO:
        nonlocal error_log_file
        if error_log_file is None:
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            error_log_file = error_log_path.open("a", encoding="utf-8")
            header = datetime.now().isoformat(timespec="seconds")
            error_log_file.write(f"\n==== {header} ====\n")
            error_log_file.flush()
        return error_log_file

    try:
        for scene_idx, scene_id in enumerate(scene_ids, start=1):
            print(f"[{scene_idx}/{total_scenes}] 处理场景 {scene_id}", flush=True)
            dataset_dir = SCENES_DIR / scene_id
            scene_task_root = TASK_OUTPUT_DIR / scene_id
            label_dir = resolve_label_directory(scene_task_root)

            if not dataset_dir.is_dir():
                print(f"⚠️  场景目录不存在：{dataset_dir}，跳过。")
                continue
            if label_dir is None:
                print(f"⚠️  在 {scene_task_root} 下未找到任何 label JSON，跳过。")
                continue

            meta = load_occupancy_metadata(dataset_dir)
            ply_path = find_ply_file(dataset_dir)
            processed_scenes += 1

            if debug_enabled:
                if label_dir == scene_task_root:
                    print(f"[DEBUG] 使用 {scene_task_root} 作为 label 配置目录。", flush=True)
                print(f"[DEBUG] Loading Gaussian model from: {ply_path}", flush=True)
            print(f"  加载点云：{ply_path.name}", flush=True)
            gaussians = GaussianModel(sh_degree=3)
            gaussians.load_ply(str(ply_path))
            if verbose_enabled:
                base_points = int(gaussians.get_xyz.shape[0])
                _log_vram_usage(
                    f"Scene {scene_id}: loaded base Gaussian model ({base_points} gaussians)",
                    device,
                )

            json_files = sorted(label_dir.glob("*.json"))
            if label_filter:
                json_files = [path for path in json_files if path.stem in label_filter]
            if not json_files:
                print(f"  没有符合条件的 label path。")
                continue

            if args.max_labels is not None:
                json_files = json_files[: args.max_labels]

            total_labels = len(json_files)
            overall_labels += total_labels
            print(f"  待渲染 {total_labels} 条路径。", flush=True)
            for label_idx, json_path in enumerate(json_files, start=1):
                print(f"    [{label_idx}/{total_labels}] → {json_path.stem}", flush=True)
                try:
                    render_path_frames(
                        scene_id=scene_id,
                        json_path=json_path,
                        gaussians=gaussians,
                        pipeline=pipeline,
                        device=device,
                        bg_color=bg_color,
                        meta=meta,
                        output_dir=args.output_dir,
                        stride=max(1, args.stride),
                        height_offset=args.height_offset,
                        look_ahead=args.look_ahead,
                        look_down=args.look_down,
                        width=width,
                        height=height,
                        fov_deg=args.fov_deg,
                        znear=args.znear,
                        zfar=args.zfar,
                        overwrite=args.overwrite,
                        view_mode=args.view_mode,
                        swap_xy=swap_xy_enabled,
                        stabilize=stabilize_enabled,
                        video=video_enabled,
                        debug=debug_enabled,
                        mirror_translation=not args.no_mirror_translation,
                        actor_runtime=actor_runtime,
                        verbose=verbose_enabled,
                        hide_actor_enabled=hide_actor_enabled,
                        render_bev=args.show_BEV,
                        mirror_bev_x=args.bev_mirror_x,
                        mirror_bev_y=args.bev_mirror_y,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"      ⚠️  渲染 {json_path.name} 失败：{exc}")
                    log_file = ensure_log_file()
                    error_count += 1
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    log_file.write(
                        f"[{timestamp}] Scene={scene_id} Label={json_path.name} Error={exc}\n"
                    )
                    log_file.write(traceback.format_exc())
                    log_file.write("\n")
                    log_file.flush()

            print(f"  场景 {scene_id} 处理完毕。\n", flush=True)
    finally:
        if error_log_file is not None:
            error_log_file.close()

    if error_count > 0:
        print(f"处理完成，但共有 {error_count} 个任务失败，详情见 {error_log_path}", flush=True)
    else:
        print("全部任务完成，未检测到错误。", flush=True)

    print(
        f"统计：实际处理 {processed_scenes} 个场景，共计 {overall_labels} 条 label 路径。",
        flush=True,
    )


if __name__ == "__main__":
    main()
