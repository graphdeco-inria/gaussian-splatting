#!/usr/bin/env python3
#search for "test" for comments to be remvoed after testing of the implementation is done

"""Render walkthrough frames for raster_world trajectories.

Debug NPC BEV-only (unmirrored, strict free-space, auto clearance):
$ python render_label_paths.py --npc-bev-debug-only \
    --tasks-dir data/selected_65k --scene 0001_839920 \
    --npc-density-coverage 0.2 --npc-count 8 --npc-priority coverage \
    --npc-density-mode angular --npc-zone-ratio 1:2:1 \
    --npc-max-range 10 --npc-free-threshold 250 --npc-free-white \
    --npc-auto-clearance --npc-actor-root ./data/human_gs_source --npc-radius-samples 80 \
    --no-mirror-translation --no-bev-mirror-x --no-bev-mirror-y \
    --npc-seed 12345 --output-dir /tmp/npc_debug

Formal render example (with NPC BEV overlays alongside full renders):
  python render_label_paths.py --tasks-dir data/task_outputs_10w_4 --scene 0001_839920 --actor-seq-dir ./data/human_gs_source/<actor_id> --npc-bev-debug --npc-density-coverage 0.2 --npc-count 4 --npc-priority coverage --npc-density-mode angular --npc-zone-ratio 1:2:1 --npc-max-range 5 --npc-seed 12345 --output-dir ./data/path_video_frames_10w_4

This utility pairs scene reconstructions under ``data/scenes`` with task
descriptions in ``data/task_outputs_10w``. For each label-path JSON that
contains a ``raster_world`` polyline, the script places a perspective camera on
that path, orients it along the motion direction, and renders individual frames
that can later be stitched into a video.
"""

#use stabilize to stabilize forward direction
from __future__ import annotations
import os
import shutil
import threading, queue
import hashlib
import glob

import json
import math
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Sequence, TextIO

import contextlib
import re
import traceback
from datetime import datetime, timezone
import time

import imageio.v2 as imageio
import numpy as np
import torch

from arguments import PipelineParams
from gaussian_renderer import render_or
from scene import GaussianModel
from scene.cameras import MiniCam
from plyfile import PlyData, PlyElement, PlyElementParseError

from utils import gaussian_ply_utils as ply_utils
from utils.general_utils import inverse_sigmoid
from utils.render_utils import (
    build_look_at,
    build_perspective_camera,
    deduplicate_points,
    derive_affine_transform,
    load_occupancy_metadata,
    load_raster_world_points,
    sample_points,
    world_to_pixel,
)
from utils.npc_density import (
    CameraWedge,
    NPCDensityConfig,
    load_free_space_mask,
    plan_npc_positions,
)

BASE_DIR = Path(__file__).resolve().parent
SCENES_DIR = BASE_DIR / "data" / "scenes"
# TASK_OUTPUT_DIR = BASE_DIR / "data" / "task_outputs_10w"
TASK_OUTPUT_DIR = BASE_DIR / "data" / "task_outputs_10w_4"
# DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "path_video_frames"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "path_video_frames_10w_4"
DEFAULT_ERROR_LOG = BASE_DIR / "10w_4_error.log"
EPS = 1e-6
FORWARD_SMOOTH_BLEND = 0.35
DEFAULT_VIDEO_FPS = 10
DEFAULT_ACTOR_PATTERN = "*.ply"
DEFAULT_ACTOR_SPEED = 1.3
ACTOR_REGION_MARGIN = 6.0
STABILIZE_WINDOW = 5
DEBUG_PLY_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("scale_x", np.float32),
        ("scale_y", np.float32),
        ("scale_z", np.float32),
        ("opacity", np.float32),
        ("actor", np.uint8),
        ("r", np.uint8),
        ("g", np.uint8),
        ("b", np.uint8),
    ]
)


def ensure_output_dir_writable(path: Path) -> None:
    """Ensure the target directory exists and is writable, raising a loud error otherwise."""

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Unable to create output directory {path}: {exc}") from exc
    sentinel = path / f".write_test_{os.getpid()}_{time.time_ns()}"
    try:
        sentinel.write_text("navdp_datagen_probe", encoding="utf-8")
    except Exception as exc:  # pylint: disable=broad-except
        raise PermissionError(f"Cannot write to output directory {path}: {exc}") from exc
    finally:
        with contextlib.suppress(Exception):
            sentinel.unlink()


###
def _disk_free_bytes(path: Path) -> int:
    """Return free bytes on the filesystem hosting 'path'."""
    usage = shutil.disk_usage(str(path))
    return int(usage.free)

def _safe_move(src: Path, dst: Path) -> None:
    """
    Cross-filesystem safe move with overwrite.
    If dst exists, remove it first to avoid merge errors.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))

class _OffloadWorker:
    """Async mover to NAS so rendering never blocks."""
    def __init__(self, maxsize: int = 256, verbose: bool = False):
        self.q: "queue.Queue[tuple[callable, tuple, dict]]" = queue.Queue(maxsize=maxsize)
        self._ok = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._verbose = verbose
        self._t.start()

    def _loop(self):
        while True:
            fn, args, kwargs = self.q.get()
            if fn is None:
                break
            try:
                fn(*args, **kwargs)
            except Exception as e:
                # mark worker unhealthy but do not crash the render loop
                self._ok = False
                if self._verbose:
                    print(f"[OFFLOAD][ASYNC] ERROR: {e}", flush=True)
            finally:
                self.q.task_done()

    def healthy(self) -> bool:
        return self._ok

    def enqueue(self, fn, *args, **kwargs):
        if not self._ok:
            # if worker is unhealthy, do the move synchronously (safe fallback)
            return fn(*args, **kwargs)
        try:
            self.q.put_nowait((fn, args, kwargs))
        except queue.Full:
            # backpressure: do it inline rather than block rendering
            fn(*args, **kwargs)

    def flush_and_stop(self):
        try:
            self.q.join()
        except Exception:
            pass
        try:
            self.q.put((None, (), {}))
        except Exception:
            pass

def offload_label_outputs(
    *, local_out_root: Path, nas_out_root: Path, scene_id: str, label_stem: str, verbose: bool = False
) -> list[tuple[Path, Path]]:
    """
    Move the just-produced outputs for one label path to the NAS mirror:
    - frames folder:   {local}/{scene}/{label_stem}/        (contains PNG, depth NPY, camera JSON)
    - ALL files starting with {label_stem}:
      * {label_stem}.mp4              (video)
      * {label_stem}_BEV.png          (bird's eye view)
      * {label_stem}_follow_path.json (metadata)
      * {label_stem}.*                (any other files)
    Returns list of (src,dst) moved entries.
    """
    moved: list[tuple[Path, Path]] = []
    local_scene_dir = local_out_root / scene_id
    nas_scene_dir   = nas_out_root   / scene_id

    # 1) frames directory (contains all frame PNGs, depth NPYs, camera JSONs)
    frames_dir = local_scene_dir / label_stem
    if frames_dir.exists() and frames_dir.is_dir():
        _safe_move(frames_dir, nas_scene_dir / label_stem)
        moved.append((frames_dir, nas_scene_dir / label_stem))
        if verbose: print(f"[OFFLOAD] moved frames dir -> {nas_scene_dir/label_stem}", flush=True)

    # 2) ALL files matching label_stem pattern
    if local_scene_dir.exists():
        for entry in local_scene_dir.iterdir():
            if entry.is_file() and entry.name.startswith(label_stem):
                dst = nas_scene_dir / entry.name
                try:
                    _safe_move(entry, dst)
                    moved.append((entry, dst))
                    if verbose: 
                        print(f"[OFFLOAD] moved {entry.name} -> {dst}", flush=True)
                except Exception as e:
                    if verbose:
                        print(f"[OFFLOAD] WARN: failed to move {entry} -> {dst}: {e}", flush=True)

    return moved

def maybe_offload_if_low_space(
    *, check_path: Path, min_free_bytes: int, local_out_root: Path, nas_out_root: Path,
    scene_id: str, label_stem: str, verbose: bool = False, offloader: _OffloadWorker | None=None
) -> None:
    """
    If free space on the filesystem hosting 'check_path' is below 'min_free_bytes',
    offload this label's outputs to NAS to free space.
    """
    free_now = _disk_free_bytes(check_path)
    if free_now < min_free_bytes:
        if verbose:
            need = _format_bytes(min_free_bytes)
            have = _format_bytes(free_now)
            print(
                f"[OFFLOAD] low free space (have {have} < need {need}); offloading {scene_id}/{label_stem} ...",
                flush=True,
            )
        if offloader is not None:
            offloader.enqueue(offload_label_outputs,
                              local_out_root=local_out_root,
                              nas_out_root=nas_out_root,
                              scene_id=scene_id,
                              label_stem=label_stem,
                              verbose=verbose)
        else:
            offload_label_outputs(
                local_out_root=local_out_root,
                nas_out_root=nas_out_root,
                scene_id=scene_id,
                label_stem=label_stem,
                verbose=verbose,
            )

def consolidate_outputs_to_nas(
    *, local_out_root: Path, nas_out_root: Path, verbose: bool = False
) -> None:
    """
    Move ALL remaining outputs under local_out_root to nas_out_root.
    Skips only temp dirs starting with '__tmp'.
    Preserves structure:
      {root}/{scene}/{label_dir}/          (frames, depth, camera JSONs)
      {root}/{scene}/{label}.mp4           (video)
      {root}/{scene}/{label}_BEV.png       (bird's eye view)
      {root}/{scene}/{label}_follow_path.json  (metadata)
      {root}/{scene}/*                     (any other files)
    """
    if not local_out_root.exists():
        return
    for scene_dir in sorted(local_out_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        nas_scene_dir = nas_out_root / scene_id

        # 1) Move ALL directories (exclude temp)
        for entry in sorted(scene_dir.iterdir()):
            if entry.is_dir():
                if entry.name.startswith("__tmp"):
                    if verbose:
                        print(f"[CONSOLIDATE] skipping temp dir {scene_id}/{entry.name}", flush=True)
                    continue
                dst = nas_scene_dir / entry.name
                try:
                    _safe_move(entry, dst)
                    if verbose:
                        print(f"[CONSOLIDATE] moved dir {scene_id}/{entry.name} -> {dst}", flush=True)
                except Exception as e:
                    print(f"[CONSOLIDATE] WARN: failed to move {entry} -> {dst}: {e}", flush=True)

        # 2) Move ALL remaining files (no filtering by extension)
        for entry in sorted(scene_dir.iterdir()):
            if entry.is_file():
                dst = nas_scene_dir / entry.name
                try:
                    _safe_move(entry, dst)
                    if verbose:
                        print(f"[CONSOLIDATE] moved file {scene_id}/{entry.name} -> {dst}", flush=True)
                except Exception as e:
                    print(f"[CONSOLIDATE] WARN: failed to move {entry} -> {dst}: {e}", flush=True)

        # 3) Clean empty scene dir
        try:
            if not any(scene_dir.iterdir()):
                scene_dir.rmdir()
                if verbose:
                    print(f"[CONSOLIDATE] removed empty scene dir {scene_id}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[CONSOLIDATE] could not remove scene dir {scene_id}: {e}", flush=True)

### helpers
def _format_bytes(num_bytes: int) -> str:
    """
    Human readable formatting for byte counts.
    """
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


def _scene_prefix(scene_id: str) -> str:
    """Return the numeric prefix before the first '_' in a scene identifier."""

    return scene_id.split("_", 1)[0]


def _log_vram_usage(message: str, device: torch.device, before_bytes: int | None = None) -> int:
    """
    Print current and delta GPU memory usage when verbose mode is active.
    Returns the current allocation so callers can reuse it as the next 'before'.
    """
    current = torch.cuda.memory_allocated(device)
    if before_bytes is None:
        delta_str = ""
    else:
        delta = current - before_bytes
        delta_str = f" (delta={_format_bytes(delta)})"
    print(f"[VERBOSE][VRAM] {message}: total={_format_bytes(current)}{delta_str}", flush=True)
    return current


class PathMetricRecorder:
    """Track per-path runtime metrics for downstream progress reporting."""

    VIDEO_STAGE = "mp4_write_sec"
    PNG_STAGE = "perframe_png_sec"
    DEPTH_STAGE = "perframe_depth_sec"
    PLY_STAGE = "ply_write_sec"

    def __init__(self, device: torch.device | None):
        self.device = device
        self.stage_seconds: defaultdict[str, float] = defaultdict(float)
        self.vram_samples: list[int] = []
        self._have_cuda = device is not None and torch.cuda.is_available()
        if self._have_cuda:
            torch.cuda.reset_peak_memory_stats(device)  # fresh peak per path

    @contextlib.contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[stage] += time.perf_counter() - start

    def sample_vram(self) -> None:
        if not self._have_cuda:
            return
        try:
            self.vram_samples.append(int(torch.cuda.memory_allocated(self.device)))
        except Exception:
            pass

    def finalize(
        self,
        *,
        scene_id: str,
        label_id: str,
        frames_rendered: int,
        total_duration: float,
        video_enabled: bool,
        job_slot: int | None,
        job_actor_id: str | None,
        job_name: str | None,
    ) -> dict:
        fps = (frames_rendered / total_duration) if frames_rendered > 0 and total_duration > 0 else None
        peak_bytes = int(torch.cuda.max_memory_allocated(self.device)) if self._have_cuda else None
        avg_bytes = None
        if self.vram_samples:
            avg_bytes = sum(self.vram_samples) / len(self.vram_samples)
        elif self._have_cuda:
            try:
                avg_bytes = float(torch.cuda.memory_allocated(self.device))
            except Exception:
                avg_bytes = None
        total_measured = sum(
            self.stage_seconds.get(stage, 0.0)
            for stage in (self.VIDEO_STAGE, self.PNG_STAGE, self.DEPTH_STAGE, self.PLY_STAGE)
        )
        stage_ratios = {}
        if total_measured > 0.0:
            for stage, seconds in self.stage_seconds.items():
                if stage in (self.VIDEO_STAGE, self.PNG_STAGE, self.DEPTH_STAGE, self.PLY_STAGE):
                    stage_ratios[stage] = seconds / total_measured
        else:
            for stage in (self.VIDEO_STAGE, self.PNG_STAGE, self.DEPTH_STAGE, self.PLY_STAGE):
                stage_ratios[stage] = 0.0

        return {
            "scene_id": scene_id,
            "label_id": label_id,
            "frames": frames_rendered,
            "duration_sec": total_duration,
            "frames_per_sec": fps,
            "video_enabled": video_enabled,
            "job_slot": job_slot,
            "job_name": job_name,
            "actor_id": job_actor_id,
            "vram_peak_bytes": peak_bytes,
            "vram_avg_bytes": avg_bytes,
            "stage_seconds": dict(self.stage_seconds),
            "stage_ratios": stage_ratios,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }


@contextlib.contextmanager
def _cuda_oom_trace(label: str, device: torch.device, verbose: bool = False):
    """
    Context manager that augments CUDA OOM exceptions with additional diagnostics.
    """
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

        import traceback

        stack_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        stack_text = ''.join(stack_lines).rstrip()

        diagnostics = (
            f"{message}\n"
            f"[CUDA OOM @ {label}] "
            f"allocated={_format_bytes(allocated)} "
            f"reserved={_format_bytes(reserved)} "
            f"max_alloc={_format_bytes(max_alloc)} "
            f"max_reserved={_format_bytes(max_reserved)}\n"
            f"Traceback (most recent call last):\n{stack_text}"
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
    animation_cycle_mod: int


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
    max_points: int


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


def _extract_scaling_components_struct(
    data: np.ndarray,
    *,
    scale_names: Sequence[str] | None = None,
    uniform_scale: bool = False,
) -> np.ndarray:
    """Return scale/log-scale triples from a structured vertex array."""
    names = data.dtype.names or ()

    if scale_names:
        if uniform_scale and len(scale_names) >= 1:
            values = np.asarray(data[scale_names[0]], dtype=np.float32).reshape(-1, 1)
            return np.repeat(values, 3, axis=1)
        cols = [np.asarray(data[name], dtype=np.float32) for name in scale_names if name in names]
        if cols:
            stacked = np.stack(cols, axis=1)
            if stacked.shape[1] >= 3:
                return stacked[:, :3].astype(np.float32, copy=False)
            if stacked.shape[1] == 1:
                return np.repeat(stacked, 3, axis=1).astype(np.float32, copy=False)
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        return np.stack(
            [np.asarray(data["scale_0"], dtype=np.float32),
             np.asarray(data["scale_1"], dtype=np.float32),
             np.asarray(data["scale_2"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
    if {"scales_0", "scales_1", "scales_2"}.issubset(names):
        return np.stack(
            [np.asarray(data["scales_0"], dtype=np.float32),
             np.asarray(data["scales_1"], dtype=np.float32),
             np.asarray(data["scales_2"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
    if "scale" in names:
        values = np.asarray(data["scale"], dtype=np.float32).reshape(-1, 1)
        return np.repeat(values, 3, axis=1)
    return np.zeros((data.shape[0], 3), dtype=np.float32)


def _build_scene_debug_vertices(data: np.ndarray) -> np.ndarray:
    """Construct a structured array representing the cropped scene gaussians."""
    if data.size == 0:
        return np.zeros((0,), dtype=DEBUG_PLY_DTYPE)

    xyz = np.stack(
        (np.asarray(data["x"], dtype=np.float32),
         np.asarray(data["y"], dtype=np.float32),
         np.asarray(data["z"], dtype=np.float32)),
        axis=1,
    )
    scaling = _extract_scaling_components_struct(data)
    opacity = (
        np.asarray(data["opacity"], dtype=np.float32).reshape(-1)
        if "opacity" in (data.dtype.names or ())
        else np.zeros(data.shape[0], dtype=np.float32)
    )

    out = np.zeros(data.shape[0], dtype=DEBUG_PLY_DTYPE)
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    out["scale_x"] = scaling[:, 0]
    out["scale_y"] = scaling[:, 1]
    out["scale_z"] = scaling[:, 2]
    out["opacity"] = opacity
    out["actor"] = 0
    out["r"] = 160
    out["g"] = 160
    out["b"] = 160
    return out


def _build_actor_debug_vertices(actor_data: np.ndarray, sequence: ActorSequence) -> np.ndarray:
    """Construct a structured array representing the transformed actor gaussians."""
    xyz = np.stack((actor_data["x"], actor_data["y"], actor_data["z"]), axis=1).astype(np.float32)
    scaling = _extract_scaling_components_struct(
        actor_data,
        scale_names=sequence.scale_names,
        uniform_scale=sequence.uniform_scale,
    )
    opacity = np.asarray(actor_data["opacity"], dtype=np.float32).reshape(-1)

    out = np.zeros(actor_data.shape[0], dtype=DEBUG_PLY_DTYPE)
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    out["scale_x"] = scaling[:, 0]
    out["scale_y"] = scaling[:, 1]
    out["scale_z"] = scaling[:, 2]
    out["opacity"] = opacity
    out["actor"] = 1
    out["r"] = 230
    out["g"] = 80
    out["b"] = 60
    return out


def _write_debug_ply(path: Path, data: np.ndarray) -> None:
    """Write the structured debug vertices to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(data, "vertex")
    PlyData([element], text=False).write(str(path))




@dataclass(frozen=True)
class ActorRuntime:
    options: ActorOptions
    sequence: ActorSequence


@dataclass(frozen=True)
class PreparedPath:
    path_xy: list[np.ndarray]
    raw_points: list[np.ndarray]
    sampled_xy: list[np.ndarray]
    affine: tuple[float, float, float, float]
    floor_z: float
    ceiling: float


@dataclass(frozen=True)
class ActorDumpOptions:
    directory: Path | None
    stride: int
    include_scene: bool
    max_frames: int | None
    dump_only: bool


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


### helper for BEV showing ###
# ===== BEV helpers =====

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        # drop alpha
        return img[..., :3]
    return img

def _blend_pixel(img: np.ndarray, u: int, v: int, color: tuple[int, int, int], alpha: float | None) -> None:
    """Blend a pixel with optional alpha; defaults to overwrite when alpha is None/1."""
    if alpha is None or alpha >= 1.0:
        img[v, u, :] = color
        return
    if alpha <= 0.0:
        return
    orig = img[v, u, :].astype(np.float32)
    target = np.array(color, dtype=np.float32)
    blended = orig * (1.0 - alpha) + target * alpha
    img[v, u, :] = blended.astype(img.dtype)

def _draw_disk(
    img: np.ndarray,
    uv: tuple[int, int],
    r: int,
    color: tuple[int, int, int],
    alpha: float | None = None,
) -> None:
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
                _blend_pixel(img, u, v, color, alpha)

def _draw_polyline(
    img: np.ndarray,
    pts: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 1,
    dotted: bool = False,
    dot_gap: int = 6,
    alpha: float | None = None,
) -> None:
    """
    Very lightweight line rasterization: sample along each segment with max(|dx|,|dy|)+1 points.
    """
    if len(pts) < 2:
        return
    h, w = img.shape[:2]

    def put(u, v):
        if 0 <= v < h and 0 <= u < w:
            _blend_pixel(img, u, v, color, alpha)
            if thickness > 1:
                for dv in range(-thickness + 1, thickness):
                    for du in range(-thickness + 1, thickness):
                        vv, uu = v + dv, u + du
                        if 0 <= vv < h and 0 <= uu < w:
                            _blend_pixel(img, uu, vv, color, alpha)

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
                continue  # skip to create dotted appearance
            u = int(round(u0 + du * (t / steps)))
            v = int(round(v0 + dv * (t / steps)))
            put(u, v)

def _draw_text_lines(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 8)) -> None:
    """
    Overlay text if Pillow is available; otherwise draw small colored ticks as a fallback.
    """
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
        # Minimal fallback: draw short magenta/green bars to indicate text lines count
        x, y = origin
        for _ in lines:
            for dx in range(40):
                if y < img.shape[0] and x + dx < img.shape[1]:
                    img[y, x + dx, :] = (200, 200, 200)
            y += 6
    # Mirror the PATHS (not the image) if requested

def _mirror_uv(uv: tuple[int, int], w: int, h: int, mirror_x: bool, mirror_y: bool) -> tuple[int, int]:
    u, v = uv
    uu = (w - 1 - u) if mirror_x else u
    vv = (h - 1 - v) if mirror_y else v
    return uu, vv

def _meters_to_px(meta: dict, meters: float) -> int:
    """Convert meters to occupancy pixel units (rounded)."""
    if meters <= 0.0:
        return 0
    return int(round(meters / float(meta["scale"])))

def _rotate_xy(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y = float(vec[0]), float(vec[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)

def _draw_circle_outline(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float | None = None,
) -> None:
    """Approximate a circle outline by drawing two filled discs."""
    if radius <= 0:
        return
    h, w = img.shape[:2]
    u, v = center
    if not (0 <= u < w and 0 <= v < h):
        return
    base_color = tuple(int(c) for c in img[v, u, :])
    _draw_disk(img, center, radius, color, alpha)
    inner = max(0, radius - 1)
    if inner > 0:
        # Restore interior so the center marker remains visible
        _draw_disk(img, center, inner, base_color, None)

def _draw_wedge(
    img: np.ndarray,
    meta: dict,
    origin_xy: np.ndarray,
    forward_xy: np.ndarray,
    fov_deg: float,
    r_min: float,
    r_max: float,
    color: tuple[int, int, int] = (255, 165, 0),
    mirror_x: bool = True,
    mirror_y: bool = True,
    alpha: float | None = None,
) -> None:
    """Draw wedge bounds (ground-plane FOV slice) and optional inner radius."""
    if r_max <= 0.0:
        return
    fov_rad = math.radians(fov_deg)
    left_dir = _rotate_xy(forward_xy, -0.5 * fov_rad)
    right_dir = _rotate_xy(forward_xy, 0.5 * fov_rad)
    left_pt = origin_xy + left_dir * r_max
    right_pt = origin_xy + right_dir * r_max
    origin_px = world_to_pixel(meta, origin_xy)
    left_px = world_to_pixel(meta, left_pt)
    right_px = world_to_pixel(meta, right_pt)
    h = int((meta["top"] - meta["bottom"]) / meta["scale"])
    w = int((meta["right"] - meta["left"]) / meta["scale"])
    origin_px = _mirror_uv(origin_px, w, h, mirror_x, mirror_y)
    left_px = _mirror_uv(left_px, w, h, mirror_x, mirror_y)
    right_px = _mirror_uv(right_px, w, h, mirror_x, mirror_y)
    _draw_polyline(img, [origin_px, left_px], color, thickness=1, dotted=False, alpha=alpha)
    _draw_polyline(img, [origin_px, right_px], color, thickness=1, dotted=False, alpha=alpha)
    _draw_polyline(img, [left_px, right_px], color, thickness=1, dotted=True, dot_gap=4, alpha=alpha)
    if r_min > 0.0 and r_min < r_max:
        radius_px = _meters_to_px(meta, r_min)
        _draw_circle_outline(img, origin_px, radius_px, color, alpha)

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

    """
    Draw a BEV (bird's-eye view) image using occupancy.png as background.
    Camera path = magenta, Actor path = green. Mark start/end.
    """
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
        # Create a blank canvas if occupancy is missing
        w = int((meta["right"] - meta["left"]) / meta["scale"])
        h = int((meta["top"] - meta["bottom"]) / meta["scale"])
        base = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        base = imageio.imread(occ_path)
        base = _ensure_rgb(base.copy())
    h, w = base.shape[:2]

    cam_pts = []
    act_pts = []
    if camera_xy_seq:
        for xy in camera_xy_seq:
            cam_pts.append(world_to_pixel(meta, xy))
    if actor_xy_seq:
        for xy in actor_xy_seq:
            act_pts.append(world_to_pixel(meta, xy))
    cam_pts = _mirror_pts(cam_pts)
    act_pts = _mirror_pts(act_pts)
    # Draw polylines
    if cam_pts:
        _draw_polyline(base, cam_pts, (255, 0, 255), thickness=2, dotted = False)  # magenta
        # start/end markers
        _draw_disk(base, cam_pts[0], 4, (255, 255, 0))            # yellow start for camera
        _draw_disk(base, cam_pts[-1], 4, (255, 0, 0))             # red end for camera
    if act_pts:
        _draw_polyline(base, act_pts, (0, 255, 0), thickness=2, dotted = actor_path_dotted)   # green
        _draw_disk(base, act_pts[0], 4, (0, 255, 255))            # cyan start for actor
        _draw_disk(base, act_pts[-1], 4, (0, 128, 0))             # dark green end for actor

    # Text overlay
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
    # base = np.rot90(base,k=2) #rotate 180 to match BEV

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, base)

def save_npc_bev_debug_image(
    *,
    scene_id: str,
    label_id: str,
    frame_idx: int,
    meta: dict,
    occ_image: np.ndarray,
    path_xy: list[np.ndarray],
    camera_xy: np.ndarray,
    forward_xy: np.ndarray,
    npc_positions: list[np.ndarray],
    config: NPCDensityConfig,
    fov_deg: float,
    wedge_max_range: float,
    achieved_coverage: float,
    requested_count: int,
    shortfall: int,
    target_coverage: float | None,
    mirror_bev_x: bool,
    mirror_bev_y: bool,
    out_path: Path,
) -> None:
    """Draw an NPC placement debug frame (camera FOV wedge + NPC discs) on occupancy."""

    occ_rgb = occ_image.copy()
    h, w = occ_rgb.shape[:2]
    pad_w = 240
    base = np.zeros((h, w + pad_w, 3), dtype=np.uint8)
    base[:, :w, :] = occ_rgb

    path_px = [_mirror_uv(world_to_pixel(meta, xy), w, h, mirror_bev_x, mirror_bev_y) for xy in path_xy]
    cam_px = _mirror_uv(world_to_pixel(meta, camera_xy), w, h, mirror_bev_x, mirror_bev_y)
    npc_px = [_mirror_uv(world_to_pixel(meta, xy), w, h, mirror_bev_x, mirror_bev_y) for xy in npc_positions]

    # Paths
    if path_px:
        _draw_polyline(base, path_px, (255, 0, 255), thickness=1, dotted=False)
        _draw_disk(base, path_px[0], 4, (255, 255, 0))
        _draw_disk(base, path_px[-1], 4, (255, 0, 0))

    # Camera marker and wedge
    _draw_disk(base, cam_px, 4, (0, 191, 255))  # camera center
    _draw_wedge(
        base,
        meta=meta,
        origin_xy=camera_xy,
        forward_xy=forward_xy,
        fov_deg=fov_deg,
        r_min=config.min_distance_from_camera,
        r_max=wedge_max_range,
        color=(255, 165, 0),
        mirror_x=mirror_bev_x,
        mirror_y=mirror_bev_y,
        alpha=0.5,
    )

    # NPC discs
    npc_radius_px = max(1, _meters_to_px(meta, config.clearance_radius))
    for xy_px in npc_px:
        _draw_disk(base, xy_px, npc_radius_px, (0, 255, 0))
        _draw_disk(base, xy_px, max(1, npc_radius_px // 3), (0, 128, 0))

    lines = [
        f"Scene: {scene_id}",
        f"Label: {label_id}",
        f"Frame: {frame_idx}",
        f"Coverage: {achieved_coverage:.2f}"
        + (f"/{target_coverage:.2f}" if target_coverage is not None else "")
        + f" ({config.coverage_mode})",
        f"NPCs: {len(npc_positions)}/{requested_count} (shortfall {shortfall})",
        f"Priority: {config.priority} | desired={config.desired_count or '-'} | max={config.max_npcs or '-'}",
        f"Zones (near:mid:far): {config.zone_weights[0]:.1f}:{config.zone_weights[1]:.1f}:{config.zone_weights[2]:.1f}",
        f"Clearance: {config.clearance_radius:.2f} m",
        f"Range: {config.min_distance_from_camera:.2f}-{wedge_max_range:.2f} m",
        f"FOV: {fov_deg:.1f} deg",
    ]

    _draw_text_lines(base, lines, origin=(w + 12, 8))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, base)

def generate_npc_bev_debug_for_path(
    *,
    scene_id: str,
    label_id: str,
    path_xy: list[np.ndarray],
    meta: dict,
    occ_image: np.ndarray,
    free_mask: np.ndarray,
    config: NPCDensityConfig,
    fov_deg: float,
    seed_base: int,
    output_root: Path,
    stabilize: bool,
    forward_fn,
    goal_xy: np.ndarray | None = None,
    npc_max_range: float | None = None,
    mirror_bev_x: bool = True,
    mirror_bev_y: bool = True,
    exclude_discs: list[tuple[np.ndarray, float]] | None = None,
) -> dict:
    """Generate per-frame NPC BEV debug images without rendering RGB/depth."""

    direction_window = STABILIZE_WINDOW if stabilize else 1
    positions = [np.array([xy[0], xy[1], 0.0], dtype=np.float32) for xy in path_xy]
    goal_xy = goal_xy if goal_xy is not None else (path_xy[-1] if path_xy else None)
    out_dir = output_root / scene_id / label_id / "__npc_bev_debug"

    wedge_max_range = npc_max_range if npc_max_range is not None else config.max_distance_from_camera
    if wedge_max_range is None:
        wedge_max_range = 5.0

    frames_done = 0
    total_shortfall = 0
    coverages: list[float] = []
    for frame_idx, cam_pos in enumerate(positions):
        forward = forward_fn(positions, frame_idx, window=direction_window)
        forward_xy = forward[:2]
        norm = float(np.linalg.norm(forward_xy))
        if norm < EPS:
            forward_xy = np.array([0.0, 1.0], dtype=np.float32)
        else:
            forward_xy = (forward_xy / norm).astype(np.float32)

        wedge = CameraWedge(
            origin_xy=cam_pos[:2],
            forward_xy=forward_xy,
            fov_deg=fov_deg,
            max_range=wedge_max_range,
            min_range=config.min_distance_from_camera,
        )
        seed = _npc_frame_seed(seed_base, scene_id, label_id, frame_idx)
        rng = np.random.default_rng(seed)
        placement = plan_npc_positions(
            wedge=wedge,
            free_mask=free_mask,
            meta=meta,
            rng=rng,
            config=config,
            goal_xy=goal_xy,
            exclude_discs=exclude_discs,
        )

        out_path = out_dir / f"npc_bev_{frame_idx:04d}.png"
        save_npc_bev_debug_image(
            scene_id=scene_id,
            label_id=label_id,
            frame_idx=frame_idx,
            meta=meta,
            occ_image=occ_image,
            path_xy=path_xy,
            camera_xy=cam_pos[:2],
            forward_xy=forward_xy,
            npc_positions=placement.positions_xy,
            config=config,
            fov_deg=fov_deg,
            wedge_max_range=wedge_max_range,
            achieved_coverage=placement.achieved_coverage,
            requested_count=placement.requested_count,
            shortfall=placement.shortfall,
            target_coverage=config.target_coverage,
            mirror_bev_x=mirror_bev_x,
            mirror_bev_y=mirror_bev_y,
            out_path=out_path,
        )
        frames_done += 1
        total_shortfall += placement.shortfall
        coverages.append(float(placement.achieved_coverage))

    coverage_min = min(coverages) if coverages else 0.0
    coverage_max = max(coverages) if coverages else 0.0
    coverage_mean = float(sum(coverages) / len(coverages)) if coverages else 0.0

    return {
        "frames": frames_done,
        "shortfall": total_shortfall,
        "output_dir": out_dir,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "coverage_mean": coverage_mean,
    }
##############################################################################################

def _serialize_camera(camera: MiniCam | OrthoMiniCam, *, orthographic: bool) -> dict:
    """
    Serialize camera parameters (intrinsics + extrinsics) to a JSON-friendly dict.
    """
    w = int(camera.image_width)
    h = int(camera.image_height)
    world_to_camera = camera.world_view_transform.detach().cpu().numpy()
    camera_to_world = torch.inverse(camera.world_view_transform).detach().cpu().numpy()
    proj = camera.full_proj_transform.detach().cpu().numpy()

    cam_type = "orthographic" if orthographic else "perspective"
    fx = fy = None
    cx = cy = None
    ortho_half_w = ortho_half_h = None
    if not orthographic:
        fx = w / (2.0 * math.tan(camera.FoVx * 0.5))
        fy = h / (2.0 * math.tan(camera.FoVy * 0.5))
        cx = w * 0.5
        cy = h * 0.5
    else:
        ortho_half_w = getattr(camera, "_half_width", None)
        ortho_half_h = getattr(camera, "_half_height", None)

    return {
        "type": cam_type,
        "resolution": {"width": w, "height": h},
        "fov": {
            "x_rad": float(camera.FoVx),
            "y_rad": float(camera.FoVy),
            "x_deg": math.degrees(float(camera.FoVx)),
            "y_deg": math.degrees(float(camera.FoVy)),
        },
        "znear": float(camera.znear),
        "zfar": float(camera.zfar),
        "intrinsics": {
            "fx": float(fx) if fx is not None else None,
            "fy": float(fy) if fy is not None else None,
            "cx": float(cx) if cx is not None else None,
            "cy": float(cy) if cy is not None else None,
            "half_width": float(ortho_half_w) if ortho_half_w is not None else None,
            "half_height": float(ortho_half_h) if ortho_half_h is not None else None,
        },
        "camera_center_world": camera.camera_center.detach().cpu().numpy().tolist(),
        "world_to_camera": world_to_camera.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "projection_matrix": proj.tolist(),
    }


# Depth encoding options: bit depth -> (step meters, max distance meters).
# 16-bit: 1 mm steps to 65.535 m; 12-bit: 1 mm steps to 4.095 m;
# 10-bit: 2 mm steps to 2.046 m; 8-bit: 4 cm steps to 10.2 m.
DEPTH_ENCODING_SPECS = {
    16: {"step_m": 0.001, "max_m": 0.001 * ((1 << 16) - 1)},
    12: {"step_m": 0.001, "max_m": 0.001 * ((1 << 12) - 1)},
    10: {"step_m": 0.002, "max_m": 0.002 * ((1 << 10) - 1)},
    8: {"step_m": 0.04, "max_m": 0.04 * ((1 << 8) - 1)},
}


def _quantize_depth(depth_m: np.ndarray, *, bit_depth: int) -> np.ndarray:
    spec = DEPTH_ENCODING_SPECS[bit_depth]
    step_m = spec["step_m"]
    max_m = spec["max_m"]
    depth_clipped = np.clip(depth_m, 0.0, max_m)
    quantized = np.rint(depth_clipped / step_m)
    dtype = np.uint8 if bit_depth <= 8 else np.uint16
    return quantized.astype(dtype)


def _save_depth_map(
    *,
    frames_dir: Path,
    frame_prefix: str,
    frame_idx: int,
    img_pkg: dict,
    orthographic: bool,
    depth_bit_depth: int,
    metrics: PathMetricRecorder | None = None,
) -> None:
    """Persist depth (PNG, quantized by bit depth) for one frame."""
    depth_tensor = img_pkg.get("depth")
    if depth_tensor is None:
        return
    timing_ctx = metrics.measure(PathMetricRecorder.DEPTH_STAGE) if metrics else contextlib.nullcontext()
    with timing_ctx:
        depth = depth_tensor.detach().cpu().numpy()
        if depth.ndim > 2:
            depth = np.squeeze(depth)
        # Rasterizer returns inverse depth (1 / meters); convert back so saved PNG stores meters.
        with np.errstate(divide="ignore"):
            depth = np.where(depth > 0.0, 1.0 / depth, 0.0)
        rotate_k = 1 if orthographic else 2
        depth_rot = np.rot90(depth, k=rotate_k)

        depth_quant = _quantize_depth(depth_rot, bit_depth=depth_bit_depth)
        depth_png_path = frames_dir / f"{frame_prefix}_{frame_idx:04d}_depth.png"
        imageio.imwrite(depth_png_path, depth_quant)


def _save_camera_metadata(
    *,
    frames_dir: Path,
    frame_prefix: str,
    frame_idx: int,
    camera: MiniCam | OrthoMiniCam,
    orthographic: bool,
) -> None:
    """Persist camera metadata JSON for one frame."""
    cam_json = _serialize_camera(camera, orthographic=orthographic)
    cam_json_path = frames_dir / f"{frame_prefix}_{frame_idx:04d}_camera.json"
    cam_json_path.write_text(json.dumps(cam_json, indent=2))

def _rotate_180_xy(xy: np.ndarray) -> np.ndarray:
    """Rotate 2D points by 180 degrees around origin."""
    return np.flipud(np.fliplr(xy))

def _npc_frame_seed(base_seed: int, scene_id: str, label_id: str, frame_idx: int) -> int:
    payload = f"{base_seed}:{scene_id}:{label_id}:{frame_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")

def _compute_ply_radius(ply_path: Path) -> float:
    """Estimate avatar radius from a PLY frame using 95th percentile of XY distances from median center."""
    ply = ply_utils.GaussianPly.read(ply_path)
    x = np.asarray(ply.data["x"], dtype=np.float64)
    y = np.asarray(ply.data["y"], dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return 0.0
    xy = np.stack((x, y), axis=1)
    center = np.median(xy, axis=0)
    dists = np.linalg.norm(xy - center[None, :], axis=1)
    return float(np.percentile(dists, 95))

def _compute_default_npc_clearance(actor_root: Path, sample_limit: int = 80) -> float:
    """Derive a default NPC clearance radius from HumanGS sources (trimmed mean of radii)."""
    if not actor_root.is_dir():
        raise FileNotFoundError(f"Actor source root not found: {actor_root}")
    radii: list[float] = []
    actor_dirs = sorted(p for p in actor_root.iterdir() if p.is_dir())
    for idx, actor_dir in enumerate(actor_dirs):
        if idx >= sample_limit:
            break
        ply_files = sorted(glob.glob(str(actor_dir / "*.ply")))
        if not ply_files:
            continue
        try:
            r = _compute_ply_radius(Path(ply_files[0]))
            if r > 0.0 and math.isfinite(r):
                radii.append(r)
        except Exception:
            continue
    if not radii:
        raise RuntimeError(f"No valid PLY frames found under {actor_root}")
    radii_sorted = sorted(radii)
    n = len(radii_sorted)
    drop = max(0, int(n * 0.125))
    trimmed = radii_sorted[drop : n - drop] if n - 2 * drop > 0 else radii_sorted
    return float(sum(trimmed) / len(trimmed))

def _parse_zone_ratio(text: str) -> tuple[float, float, float]:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected near:mid:far ratio like 1:2:1, got '{text}'")
    vals = []
    for part in parts:
        try:
            vals.append(float(part))
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Invalid zone ratio component '{part}' in '{text}'") from exc
    if all(v <= 0.0 for v in vals):
        return (1.0, 1.0, 1.0)
    return tuple(vals)  # type: ignore[return-value]
def build_navdp_mask_ply(
    *,
    scene_id: str,
    label_id: str,
    json_path: Path,
    meta: dict,
    gaussians: GaussianModel,
    path_xy: Sequence[np.ndarray],
    output_dir: Path,
) -> None:
    """
    Project scene Gaussians and the path onto z=0 using the scene's occupancy.png mask.
    This ensures the path is overlaid separately as black dots, not mixed into scene Gaussians.
    White pixels in the mask are left blank; all other pixels accept Gaussian dots.
    """
    mask_path = SCENES_DIR / scene_id / "occupancy.png"
    if not mask_path.is_file():
        print(
            f"      WARNING: Missing occupancy mask PNG at {mask_path}; skipping NavDP PLY.",
            flush=True,
        )
        return

    mask = imageio.imread(mask_path)
    mask = _ensure_rgb(np.array(mask))
    h, w = mask.shape[:2]

    # Scene Gaussians -> mask filter (skip white pixels)
    # Cache CPU XY to avoid repeating huge transfers per label.
    xy: np.ndarray
    if hasattr(gaussians, "_navdp_xy_cache"):
        xy = getattr(gaussians, "_navdp_xy_cache")
    else:
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        xy = xyz[:, :2]
        setattr(gaussians, "_navdp_xy_cache", xy)
    u = np.round((xy[:, 0] - float(meta["left"])) / float(meta["scale"])).astype(np.int64)
    v = np.round((float(meta["top"]) - xy[:, 1]) / float(meta["scale"])).astype(np.int64)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_valid = u[in_bounds]
    v_valid = v[in_bounds]
    xy_valid = xy[in_bounds]
    mask_samples = mask[v_valid, u_valid, :]
    is_white = np.all(mask_samples >= 250, axis=1)
    xy_kept = xy_valid[~is_white]
    xy_kept = _rotate_180_xy(xy_kept)
    # Path polyline projected to z=0 (respect mask to keep white areas blank)
    path_xy_arr = np.stack(path_xy, axis=0)
    pu = np.round((path_xy_arr[:, 0] - float(meta["left"])) / float(meta["scale"])).astype(np.int64)
    pv = np.round((float(meta["top"]) - path_xy_arr[:, 1]) / float(meta["scale"])).astype(np.int64)
    p_in_bounds = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
    pu = pu[p_in_bounds]
    pv = pv[p_in_bounds]
    path_valid = path_xy_arr[p_in_bounds]
    p_is_white = np.all(mask[pv, pu, :] >= 250, axis=1)
    path_kept = path_valid[~p_is_white]

    ply_dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ]
    )

    total = xy_kept.shape[0] + path_kept.shape[0]
    if total == 0:
        print(
            f"      WARNING: Mask rejected all points; no NavDP PLY generated for {label_id}.",
            flush=True,
        )
        return

    ply_data = np.empty(total, dtype=ply_dtype)
    if xy_kept.shape[0] > 0:
        ply_data[: xy_kept.shape[0]]["x"] = xy_kept[:, 0]
        ply_data[: xy_kept.shape[0]]["y"] = xy_kept[:, 1]
        ply_data[: xy_kept.shape[0]]["z"] = 0.0
        ply_data[: xy_kept.shape[0]][["red", "green", "blue"]] = (0, 0, 255)

    if path_kept.shape[0] > 0:
        start = xy_kept.shape[0]
        end = start + path_kept.shape[0]
        ply_data[start:end]["x"] = path_kept[:, 0]
        ply_data[start:end]["y"] = path_kept[:, 1]
        ply_data[start:end]["z"] = 0.0
        ply_data[start:end][["red", "green", "blue"]] = (0, 0, 0)

    element = PlyElement.describe(ply_data, "vertex")
    ply = PlyData([element], text=False)
    out_path = output_dir / scene_id / f"{label_id}_navdp_mask.ply"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_path))
    print(f"      NavDP mask PLY saved to {out_path}", flush=True)


class NavdpPlyCoordinator:
    """Control how often NavDP PLYs are produced (per-label or per-scene)."""

    def __init__(self, *, per_scene: bool = False):
        self.per_scene = bool(per_scene)
        self._pending_paths: dict[str, list[list[np.ndarray]]] = defaultdict(list)

    def add_path(
        self,
        *,
        scene_id: str,
        label_id: str,
        path_xy: Sequence[np.ndarray],
        meta: dict,
        gaussians: GaussianModel,
        output_dir: Path,
    ) -> None:
        if not self.per_scene:
            build_navdp_mask_ply(
                scene_id=scene_id,
                label_id=label_id,
                json_path=Path(label_id),
                meta=meta,
                gaussians=gaussians,
                path_xy=path_xy,
                output_dir=output_dir,
            )
            return
        # Cache raw coordinates for later scene-level export.
        cached: list[np.ndarray] = [np.array(pt, copy=True) for pt in path_xy]
        self._pending_paths[scene_id].append(cached)

    def finalize_scene(
        self,
        *,
        scene_id: str,
        meta: dict,
        gaussians: GaussianModel,
        output_dir: Path,
    ) -> None:
        if not self.per_scene:
            return
        bundles = self._pending_paths.pop(scene_id, [])
        if not bundles:
            return
        combined: list[np.ndarray] = []
        for bundle in bundles:
            combined.extend(bundle)
        if not combined:
            return
        build_navdp_mask_ply(
            scene_id=scene_id,
            label_id=f"{scene_id}_scene",
            json_path=Path(scene_id),
            meta=meta,
            gaussians=gaussians,
            path_xy=combined,
            output_dir=output_dir,
        )


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
    """Return sorted actor frame paths, falling back to *.ply when pattern is too strict."""

    if not options.sequence_dir.is_dir():
        return []

    pattern = options.pattern or "*.ply"
    initial = [
        path
        for path in options.sequence_dir.glob(pattern)
        if path.is_file()
    ]
    initial = [path for path in initial if path.suffix.lower() == ".ply"]

    if not initial:
        initial = [
            path
            for path in options.sequence_dir.glob("*.ply")
            if path.is_file()
    ]

    return sorted(initial, key=natural_sort_key)


def _first_actor_subdir(root: Path) -> Path | None:
    """Pick the first child directory under root that contains at least one PLY."""

    if root is None or not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.ply")):
            return child
    return None


def load_gaussian_ply(path: Path) -> ply_utils.GaussianPly:
    """Load a Gaussian PLY. Raise if parsing fails so we avoid costly GPU fallbacks."""

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
    """Load and normalise an animated actor sequence comprised of per-frame PLYs."""

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
        max_points=max(frame.base_data.shape[0] for frame in frames),
    )


def apply_transform_to_frame(
    base_frame: ActorSequenceFrame,
    sequence: ActorSequence,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transform to a stored actor frame and return the mutated vertex array."""

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
    target_rest_dim: int | None = None,
) -> ActorRenderFrame:
    """Convert transformed actor Gaussian data into torch tensors on the target device."""

    allocation_before = torch.cuda.memory_allocated(device) if verbose else None

    xyz_np = np.stack((data["x"], data["y"], data["z"]), axis=1).astype(np.float32)
    xyz = torch.from_numpy(xyz_np).to(device)

    dc_names = [f"f_dc_0", f"f_dc_1", f"f_dc_2"]
    if not all(name in data.dtype.names for name in dc_names):
        missing = [name for name in dc_names if name not in data.dtype.names]
        raise KeyError(f"Actor PLY missing DC SH coefficients: {missing}")
    features_dc_np = np.stack([data[name] for name in dc_names], axis=1).astype(np.float32)
    features_dc = torch.from_numpy(features_dc_np[:, :, None]).to(device).transpose(1, 2).contiguous()

    source_rest_dim = sequence.rest_dim if sequence.rest_dim > 0 else 0
    expected_rest_dim = int(target_rest_dim) if target_rest_dim is not None else source_rest_dim
    if expected_rest_dim < 0:
        raise ValueError("target_rest_dim must be non-negative")

    if source_rest_dim > 0:
        rest_np = np.stack(
            [data[name] for name in sequence.feature_rest_names],
            axis=1,
        ).astype(np.float32)
        rest_np = rest_np.reshape(data.shape[0], 3, sequence.rest_dim)
        features_rest_src = torch.from_numpy(rest_np.transpose(0, 2, 1)).to(device)
    else:
        features_rest_src = torch.zeros(
            (data.shape[0], 0, 3),
            dtype=torch.float32,
            device=device,
        )

    if expected_rest_dim == features_rest_src.shape[1]:
        features_rest = features_rest_src
    else:
        features_rest = torch.zeros(
            (data.shape[0], expected_rest_dim, 3),
            dtype=torch.float32,
            device=device,
        )
        copy_dim = min(features_rest_src.shape[1], expected_rest_dim)
        if copy_dim > 0:
            features_rest[:, :copy_dim] = features_rest_src[:, :copy_dim]

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


def build_marker_actor_render(
    *,
    position_xy: np.ndarray,
    floor_z: float,
    sequence: ActorSequence,
    gaussians: GaussianModel,
    device: torch.device,
    radius: float = 0.1,
) -> ActorRenderFrame:
    """Create a single-gaussian actor frame to visualize the hidden avatar position."""

    center_z = floor_z + sequence.hip_height
    xyz = torch.tensor([[float(position_xy[0]), float(position_xy[1]), float(center_z)]], device=device, dtype=torch.float32)

    features_dc = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    features_dc[0, 0, 0] = 1.0

    base_rest = gaussians.get_features_rest
    if base_rest.shape[1] > 0:
        features_rest = torch.zeros((1, base_rest.shape[1], base_rest.shape[2]), device=device, dtype=torch.float32)
    else:
        features_rest = torch.zeros((1, 0, 0), device=device, dtype=torch.float32)

    opacity = torch.full((1, 1), 0.95, device=device, dtype=torch.float32)

    base_scaling = gaussians.get_scaling
    if base_scaling.shape[1] > 0:
        scaling = torch.full((1, base_scaling.shape[1]), radius, device=device, dtype=torch.float32)
    else:
        scaling = torch.zeros((1, 0), device=device, dtype=torch.float32)

    base_rotation = gaussians.get_rotation
    rotation = torch.zeros((1, base_rotation.shape[1]), device=device, dtype=torch.float32)
    if base_rotation.shape[1] > 0:
        rotation[0, 0] = 1.0

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
        self._xyz[: self.base_size] = base._xyz.detach()
        self._xyz[self.base_size :] = actor_frame.xyz

        dc_base = base._features_dc.detach()
        self._features_dc = torch.empty(
            (self.base_size + actor_size, dc_base.shape[1], dc_base.shape[2]),
            device=device,
            dtype=dc_base.dtype,
        )
        self._features_dc[: self.base_size] = dc_base
        self._features_dc[self.base_size :] = actor_frame.features_dc

        rest_base = base._features_rest.detach()
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

        opacity_base = base._opacity.detach()
        self._opacity = torch.empty((self.base_size + actor_size, 1), device=device, dtype=opacity_base.dtype)
        self._opacity[: self.base_size] = opacity_base
        self._opacity[self.base_size :] = actor_frame.opacity

        scaling_base = base._scaling.detach()
        self._scaling = torch.empty((self.base_size + actor_size, scaling_base.shape[1]), device=device, dtype=scaling_base.dtype)
        self._scaling[: self.base_size] = scaling_base
        if actor_frame.scaling.shape[1] == 0:
            self._scaling[self.base_size :] = 0.0
        else:
            self._scaling[self.base_size :] = actor_frame.scaling

        rotation_base = base._rotation.detach()  # <- CHANGED: use _rotation not get_rotation
        self._rotation = torch.empty((self.base_size + actor_size, rotation_base.shape[1]), device=device, dtype=rotation_base.dtype)
        self._rotation[: self.base_size] = rotation_base
        self._rotation[self.base_size :] = actor_frame.rotation

        self.setup_functions()

    def setup_functions(self):
        """Copy activation functions from GaussianModel"""
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

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
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)



def _concat_actor_frames(frames: Sequence[ActorRenderFrame]) -> ActorRenderFrame:
    """Concatenate multiple ActorRenderFrame entries into a single batch."""

    if not frames:
        raise ValueError("No actor frames provided for concatenation.")
    xyz = torch.cat([f.xyz for f in frames], dim=0)
    features_dc = torch.cat([f.features_dc for f in frames], dim=0)
    features_rest = torch.cat([f.features_rest for f in frames], dim=0)
    opacity = torch.cat([f.opacity for f in frames], dim=0)
    scaling = torch.cat([f.scaling for f in frames], dim=0)
    rotation = torch.cat([f.rotation for f in frames], dim=0)
    return ActorRenderFrame(
        xyz=xyz,
        features_dc=features_dc,
        features_rest=features_rest,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
    )





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
    save_rgb_frames: bool,
    save_depth_maps: bool,
    save_camera_metadata: bool,
    depth_bit_depth: int,
    frames_dir: Path,
    frame_prefix: str,
    debug: bool,
    stabilize: bool,
    verbose: bool,
    dump: ActorDumpOptions | None = None,
    metrics: PathMetricRecorder | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-point camera walkthrough (base scene only) with actor pose preview."""

    options = actor_runtime.options
    sequence = actor_runtime.sequence
    follow_distance_m = max(float(options.follow_distance), 0.0)

    sampler = PathSampler(path_xy)
    distances = list(sampler.cumulative)
    total_length = sampler.total_length
    max_camera_distance = max(total_length - follow_distance_m, 0.0)

    frame_plans: list[ActorFramePlan] = []
    camera_positions: list[np.ndarray] = []
    actor_xy_seq_would_be: list[np.ndarray] = []
    cached_direction = np.array([0.0, 1.0], dtype=np.float32)
    prev_actor_dir: np.ndarray | None = None

    if verbose:
        print(
            f"[VERBOSE] Scene {scene_id} / {label_id}: camera-only per-point path with {len(distances)} samples.",
            flush=True,
        )

    for i, dist in enumerate(distances):
        camera_distance = min(dist, max_camera_distance)
        actor_distance = min(camera_distance + follow_distance_m, total_length)
        direction_xy = sampler.direction_at(actor_distance)
        if np.linalg.norm(direction_xy) < 1e-6:
            direction_xy = cached_direction
        else:
            cached_direction = direction_xy
        camera_xy = sampler.position_at(camera_distance)
        actor_pos_xy = sampler.position_at(actor_distance)
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

        if camera_distance >= max_camera_distance - 1e-6:
            break

    if not frame_plans:
        return [], []

    total_steps = len(frame_plans)
    direction_window = STABILIZE_WINDOW if stabilize else 1
    prev_forward: np.ndarray | None = None
    frame_counter = 0

    if metrics:
        metrics.sample_vram()

    for idx, plan in enumerate(frame_plans):
        camera_position = camera_positions[idx]

        actor_xy = actor_xy_seq_would_be[idx]
        actor_render = build_marker_actor_render(
            position_xy=actor_xy,
            floor_z=floor_z,
            sequence=sequence,
            gaussians=gaussians,
            device=device,
        )

        combined_model = CombinedGaussianModel(gaussians, actor_render)

        forward = forward_direction(camera_positions, idx, window=direction_window)
        if np.linalg.norm(forward[:2]) < EPS:
            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if stabilize and prev_forward is not None:
            blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + forward * FORWARD_SMOOTH_BLEND
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm > EPS:
                forward = (blended / blended_norm).astype(np.float32)
        prev_forward = forward.copy()

        target_xy = camera_position[:2] + forward[:2] * look_ahead
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
            f"Scene {scene_id} / {label_id}: camera-only render {idx}",
            device,
            verbose,
        ):
            img_pkg = render_or(camera, combined_model, pipeline, bg_color=bg_color, orthographic=False)
        render = img_pkg["render"].detach().cpu().numpy()
        render_uint8 = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
        render_uint8 = np.rot90(render_uint8, k=2)

        # Save requested artifacts.
        if save_rgb_frames:
            try:
                frame_path = frames_dir / f"{frame_prefix}_{frame_counter:04d}.png"
                timing_ctx = (
                    metrics.measure(PathMetricRecorder.PNG_STAGE)
                    if metrics is not None
                    else contextlib.nullcontext()
                )
                with timing_ctx:
                    imageio.imwrite(frame_path, render_uint8)
            except Exception as e:
                print(f"[WARN] Failed to save RGB frame {frame_counter}: {e}", flush=True)
        if save_depth_maps:
            try:
                _save_depth_map(
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=frame_counter,
                    img_pkg=img_pkg,
                    orthographic=False,
                    depth_bit_depth=depth_bit_depth,
                    metrics=metrics,
                )
            except Exception as e:
                print(f"[WARN] Failed to save depth for frame {frame_counter}: {e}", flush=True)
        if save_camera_metadata:
            try:
                _save_camera_metadata(
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=frame_counter,
                    camera=camera,
                    orthographic=False,
                )
            except Exception as e:
                print(f"[WARN] Failed to save camera metadata for frame {frame_counter}: {e}", flush=True)

        # Also write to video if requested
        if video:
            try:
                timing_ctx = (
                    metrics.measure(PathMetricRecorder.VIDEO_STAGE)
                    if metrics is not None
                    else contextlib.nullcontext()
                )
                with timing_ctx:
                    writer.append_data(render_uint8)
            except Exception as e:
                print(f"[ERROR] Failed to append frame {frame_counter} to video: {e}", flush=True)
                raise

        frame_counter += 1
        if metrics:
            metrics.sample_vram()

        del combined_model
        del actor_render

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
    save_rgb_frames: bool,
    save_depth_maps: bool,
    save_camera_metadata: bool,
    depth_bit_depth: int,
    frames_dir: Path,
    frame_prefix: str,
    debug: bool,
    stabilize: bool,
    verbose: bool,
    dump: ActorDumpOptions | None,
    scene_vertices: np.ndarray | None,
    scene_dtype: np.dtype | None,
    scene_template: ply_utils.GaussianPly,
    combined_tmp_dir: Path | None,
    sh_degree: int,
    gpu_only: bool,
    metrics: PathMetricRecorder | None = None,
    npc_enabled: bool = False,
    npc_config: NPCDensityConfig | None = None,
    npc_free_mask: np.ndarray | None = None,
    npc_meta: dict | None = None,
    npc_seed_base: int = 0,
    npc_wedge_max_range: float | None = None,
    npc_forward_fn=None,
    npc_goal: np.ndarray | None = None,
    npc_actor_runtime: ActorRuntime | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render combined scene+actor frames using either CPU or GPU composition."""

    options = actor_runtime.options
    sequence = actor_runtime.sequence
    if not sequence.frames:
        raise ValueError("Actor sequence is empty; cannot render.")

    sampler = PathSampler(path_xy)  
    distances = list(sampler.cumulative)
    total_length = sampler.total_length
    scene_rest_dim = int(gaussians.get_features_rest.shape[1])

    follow_distance_m = max(float(options.follow_distance), 0.0)
    max_camera_distance = max(total_length - follow_distance_m, 0.0)
    actor_ground_z = floor_z + options.foot_offset

    cycle_mod = max(1, int(getattr(options, "animation_cycle_mod", 1)))
    anim_step = (options.fps / float(DEFAULT_VIDEO_FPS)) * cycle_mod
    anim_cursor = 0.0
    num_actor_frames = len(sequence.frames)

    npc_active = (
        npc_enabled
        and npc_config is not None
        and npc_free_mask is not None
        and npc_meta is not None
        and npc_actor_runtime is not None
    )
    npc_sequence = npc_actor_runtime.sequence if npc_active else None
    npc_options = npc_actor_runtime.options if npc_active else None
    npc_ground_z = floor_z + (npc_options.foot_offset if npc_options is not None else 0.0)
    npc_cycle_mod = max(1, int(getattr(npc_options, "animation_cycle_mod", 1))) if npc_options is not None else 1
    npc_anim_step = (
        (npc_options.fps / float(DEFAULT_VIDEO_FPS)) * npc_cycle_mod if npc_options is not None else 0.0
    )
    npc_num_frames = len(npc_sequence.frames) if npc_sequence is not None else 0
    npc_shortfall_total = 0

    frame_plans: list[ActorFramePlan] = []
    camera_positions: list[np.ndarray] = []
    cached_direction = np.array([0.0, 1.0], dtype=np.float32)
    prev_actor_dir: np.ndarray | None = None

    dump_enabled = dump is not None and dump.directory is not None
    if dump and dump.dump_only and not dump_enabled:
        raise ValueError("actor_dump_only requires --actor-dump-ply-dir to be specified.")
    dump_stride = max(dump.stride, 1) if dump_enabled else 1
    dump_max = dump.max_frames if dump_enabled else None
    dump_count = 0

    if verbose:
        print(
            f"[VERBOSE] Scene {scene_id} / {label_id}: rendering actor path with {len(distances)} samples.",
            flush=True,
        )

    for i, dist in enumerate(distances):
        camera_distance = min(dist, max_camera_distance)
        actor_distance = min(camera_distance + follow_distance_m, total_length)

        direction_xy = sampler.direction_at(actor_distance)
        if np.linalg.norm(direction_xy) < 1e-6:
            direction_xy = cached_direction
        actor_dir = direction_xy.copy()
        if stabilize and prev_actor_dir is not None:
            blended_actor = prev_actor_dir * (1.0 - FORWARD_SMOOTH_BLEND) + actor_dir * FORWARD_SMOOTH_BLEND
            norm_actor = np.linalg.norm(blended_actor)
            if norm_actor > EPS:
                actor_dir = blended_actor / norm_actor
        if np.linalg.norm(direction_xy) >= 1e-6:
            cached_direction = actor_dir
        prev_actor_dir = actor_dir

        theta = math.atan2(actor_dir[0], actor_dir[1]) + math.pi
        rotation_np = rotation_matrix_z_np(theta)

        actor_pos_xy = sampler.position_at(actor_distance)
        translation_vec = np.array([actor_pos_xy[0], actor_pos_xy[1], actor_ground_z], dtype=np.float64)
        transform = build_transform_matrix(rotation_np, translation_vec)

        if options.loop:
            anim_idx = int(anim_cursor) % num_actor_frames
        else:
            anim_idx = min(int(anim_cursor), num_actor_frames - 1)
        anim_cursor += anim_step

        frame_plans.append(
            ActorFramePlan(
                base_frame_index=anim_idx,
                transform=transform,
                actor_pos_xy=np.array(actor_pos_xy, dtype=np.float32),
                direction_xy=np.array(actor_dir, dtype=np.float32),
                camera_offset=0.0,
            )
        )
        camera_xy = sampler.position_at(camera_distance)
        camera_positions.append(np.array([camera_xy[0], camera_xy[1], camera_z], dtype=np.float32))

        if camera_distance >= max_camera_distance - 1e-6:
            break

    if not frame_plans:
        return [], []

    total_steps = len(frame_plans)
    actor_xy_seq = [plan.actor_pos_xy.copy() for plan in frame_plans]
    camera_xy_seq = [pos[:2].copy() for pos in camera_positions]

    actor_xy_all = np.stack(actor_xy_seq, axis=0)
    camera_xy_all = np.stack(camera_xy_seq, axis=0)
    combined_xy = np.concatenate((actor_xy_all, camera_xy_all), axis=0)
    margin = ACTOR_REGION_MARGIN + max(look_ahead, 0.0)
    min_x = float(combined_xy[:, 0].min() - margin)
    max_x = float(combined_xy[:, 0].max() + margin)
    min_y = float(combined_xy[:, 1].min() - margin)
    max_y = float(combined_xy[:, 1].max() + margin)

    frame_counter = 0
    prev_forward: np.ndarray | None = None
    direction_window = STABILIZE_WINDOW if stabilize else 1

    if metrics:
        metrics.sample_vram()

    if gpu_only:
        # Prepare debug dumps with full-scene coverage
        scene_debug_vertices = None
        if dump_enabled and dump.include_scene:
            scene_array = np.array(scene_template.data, copy=False)
            scene_debug_vertices = _build_scene_debug_vertices(scene_array.copy())

        combined_model: CombinedGaussianModel | None = None
        combined_actor_size: int | None = None

        for idx, plan in enumerate(frame_plans):
            camera_position = camera_positions[idx]
            forward = npc_forward_fn(camera_positions, idx, window=direction_window) if npc_forward_fn else forward_direction(camera_positions, idx, window=direction_window)
            if np.linalg.norm(forward[:2]) < EPS:
                forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if stabilize and prev_forward is not None:
                blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + forward * FORWARD_SMOOTH_BLEND
                blended_norm = float(np.linalg.norm(blended))
                if blended_norm > EPS:
                    forward = (blended / blended_norm).astype(np.float32)
                    forward[2] = 0.0
            prev_forward = forward.copy()

            # Transform actor for this frame
            base_frame = sequence.frames[plan.base_frame_index]
            actor_data = apply_transform_to_frame(base_frame, sequence, plan.transform)

            npc_actor_frames: list[ActorRenderFrame] = []
            npc_debug_vertices: list[np.ndarray] = []
            if npc_active and npc_num_frames > 0 and npc_wedge_max_range is not None:
                wedge_forward = forward[:2]
                norm_fwd = float(np.linalg.norm(wedge_forward))
                if norm_fwd < EPS:
                    wedge_forward = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    wedge_forward = (wedge_forward / norm_fwd).astype(np.float32)
                wedge = CameraWedge(
                    origin_xy=camera_position[:2],
                    forward_xy=wedge_forward,
                    fov_deg=fov_deg,
                    max_range=float(npc_wedge_max_range),
                    min_range=npc_config.min_distance_from_camera,
                )
                npc_seed = _npc_frame_seed(npc_seed_base, scene_id, label_id, idx)
                rng = np.random.default_rng(npc_seed)
                placement = plan_npc_positions(
                    wedge=wedge,
                    free_mask=npc_free_mask,
                    meta=npc_meta,
                    rng=rng,
                    config=npc_config,
                    goal_xy=npc_goal,
                    exclude_discs=[(plan.actor_pos_xy, npc_config.clearance_radius)],
                )
                npc_shortfall_total += placement.shortfall
                if verbose and placement.shortfall > 0:
                    print(
                        f"[VERBOSE] NPC shortfall frame {idx}: {placement.shortfall} (requested {placement.requested_count})",
                        flush=True,
                    )
                orient_rng = np.random.default_rng(npc_seed + 1)
                for npc_i, npc_xy in enumerate(placement.positions_xy):
                    yaw = float(orient_rng.uniform(-math.pi, math.pi))
                    rotation_np = rotation_matrix_z_np(yaw)
                    translation_vec = np.array([npc_xy[0], npc_xy[1], npc_ground_z], dtype=np.float64)
                    transform = build_transform_matrix(rotation_np, translation_vec)
                    if npc_options.loop:
                        npc_anim_idx = int((idx * npc_anim_step + npc_i) % npc_num_frames)
                    else:
                        npc_anim_idx = min(int(idx * npc_anim_step + npc_i), max(npc_num_frames - 1, 0))
                    base_npc_frame = npc_sequence.frames[npc_anim_idx]
                    npc_data = apply_transform_to_frame(base_npc_frame, npc_sequence, transform)
                    if dump_enabled and (dump_max is None or dump_count < dump_max) and (idx % dump_stride == 0):
                        npc_debug_vertices.append(_build_actor_debug_vertices(npc_data, npc_sequence))
                    npc_actor_frames.append(
                        actor_data_to_tensors(
                            npc_data,
                            npc_sequence,
                            device,
                            verbose=False,
                            target_rest_dim=scene_rest_dim,
                        )
                    )

            # Dump debug PLY if requested
            if dump_enabled and (dump_max is None or dump_count < dump_max) and (idx % dump_stride == 0):
                actor_vertices = _build_actor_debug_vertices(actor_data, sequence)
                entries = []
                if scene_debug_vertices is not None and scene_debug_vertices.size > 0:
                    entries.append(scene_debug_vertices)
                entries.append(actor_vertices)
                if npc_debug_vertices:
                    entries.extend(npc_debug_vertices)
                debug_vertices = entries[0] if len(entries) == 1 else np.concatenate(entries, axis=0)
                dump_path = dump.directory / f"{frame_prefix}_{idx:04d}.ply"
                _write_debug_ply(dump_path, debug_vertices)
                dump_count += 1

            if dump and dump.dump_only:
                continue

            # Convert actor + NPCs to GPU tensors
            actor_render = actor_data_to_tensors(
                actor_data,
                sequence,
                device,
                verbose=verbose,
                target_rest_dim=scene_rest_dim,
            )
            actor_renders = [actor_render]
            if npc_actor_frames:
                actor_renders.extend(npc_actor_frames)
            merged_render = _concat_actor_frames(actor_renders) if len(actor_renders) > 1 else actor_render

            # Combine scene + actor(s)
            current_actor_size = int(merged_render.xyz.shape[0])
            if combined_model is None or combined_actor_size != current_actor_size:
                combined_model = CombinedGaussianModel(gaussians, merged_render)
                combined_actor_size = current_actor_size
            else:
                combined_model.update_actor(merged_render)

            target_xy = camera_position[:2] + forward[:2] * look_ahead
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

            # Render
            #debug
            if idx == 0 and verbose:  # Only print for first frame
                print(f"[DEBUG] Scene gaussians: {gaussians.get_xyz.shape[0]}")
                print(f"[DEBUG] Scene max_sh_degree: {gaussians.max_sh_degree}, active: {gaussians.active_sh_degree}")
                print(f"[DEBUG] Scene features_dc shape: {gaussians.get_features_dc.shape}")
                print(f"[DEBUG] Scene features_rest shape: {gaussians.get_features_rest.shape}")
                
                print(f"[DEBUG] Actor gaussians: {merged_render.xyz.shape[0]}")
                print(f"[DEBUG] Actor features_dc shape: {merged_render.features_dc.shape}")
                print(f"[DEBUG] Actor features_rest shape: {merged_render.features_rest.shape}")
                
                print(f"[DEBUG] Combined model max_sh_degree: {combined_model.max_sh_degree}, active: {combined_model.active_sh_degree}")
                print(f"[DEBUG] Combined features_dc shape: {combined_model.get_features_dc.shape}")
                print(f"[DEBUG] Combined features_rest shape: {combined_model.get_features_rest.shape}")
                print(f"[DEBUG] Combined xyz shape: {combined_model.get_xyz.shape}")
            img_pkg = render_or(camera, combined_model, pipeline, bg_color=bg_color, orthographic=False)
            render = img_pkg['render'].detach().cpu().numpy()
            render_uint8 = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
            render_uint8 = np.rot90(render_uint8, k=2)

            # Save requested artifacts without blocking rendering.
            if save_rgb_frames:
                try:
                    frame_path = frames_dir / f"{frame_prefix}_{frame_counter:04d}.png"
                    timing_ctx = (
                        metrics.measure(PathMetricRecorder.PNG_STAGE)
                        if metrics is not None
                        else contextlib.nullcontext()
                    )
                    with timing_ctx:
                        imageio.imwrite(frame_path, render_uint8)
                except Exception as e:
                    print(f"[WARN] Failed to save RGB frame {frame_counter}: {e}", flush=True)
            if save_depth_maps:
                try:
                    _save_depth_map(
                        frames_dir=frames_dir,
                        frame_prefix=frame_prefix,
                        frame_idx=frame_counter,
                        img_pkg=img_pkg,
                        orthographic=False,
                        depth_bit_depth=depth_bit_depth,
                        metrics=metrics,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to save depth for frame {frame_counter}: {e}", flush=True)
            if save_camera_metadata:
                try:
                    _save_camera_metadata(
                        frames_dir=frames_dir,
                        frame_prefix=frame_prefix,
                        frame_idx=frame_counter,
                        camera=camera,
                        orthographic=False,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to save camera metadata for frame {frame_counter}: {e}", flush=True)
            
            # Also write to video if requested (do this AFTER frame saving to avoid corruption)
            if video:
                try:
                    timing_ctx = (
                        metrics.measure(PathMetricRecorder.VIDEO_STAGE)
                        if metrics is not None
                        else contextlib.nullcontext()
                    )
                    with timing_ctx:
                        writer.append_data(render_uint8)
                except Exception as e:
                    print(f"[ERROR] Failed to append frame {frame_counter} to video: {e}", flush=True)
                    raise  # Re-raise video errors as they're critical

            frame_counter += 1
            if metrics:
                metrics.sample_vram()

            del actor_render
            torch.cuda.empty_cache()

            if verbose and (idx % 10 == 0 or idx == total_steps - 1):
                print(
                    f"[VERBOSE] Scene {scene_id} / {label_id}: frame {idx + 1}/{total_steps} complete.",
                    flush=True,
                )

        if verbose:
            print(
                f"[VERBOSE] Scene {scene_id} / {label_id}: actor rendering complete ({frame_counter} frames).",
                flush=True,
            )
        if npc_active and verbose and npc_shortfall_total > 0:
            print(
                f"[VERBOSE] Scene {scene_id} / {label_id}: NPC placement shortfall total {npc_shortfall_total}.",
                flush=True,
            )

        return camera_xy_seq, actor_xy_seq

    if scene_vertices is None or scene_dtype is None or combined_tmp_dir is None:
        raise ValueError("CPU compositor requires scene vertex data; rerun without --gpu-only.")

    scene_subset = scene_vertices.copy()

    scene_debug_vertices = (
        _build_scene_debug_vertices(scene_subset)
        if dump_enabled and dump.include_scene
        else None
    )

    combined_tmp_dir.mkdir(parents=True, exist_ok=True)
    combined_tmp_path = combined_tmp_dir / f"{label_id}_combined.ply"
    frame_gaussians = GaussianModel(sh_degree=sh_degree)

    for idx, plan in enumerate(frame_plans):
        camera_position = camera_positions[idx]
        forward = npc_forward_fn(camera_positions, idx, window=direction_window) if npc_forward_fn else forward_direction(camera_positions, idx, window=direction_window)
        if np.linalg.norm(forward[:2]) < EPS:
            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if stabilize and prev_forward is not None:
            blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + forward * FORWARD_SMOOTH_BLEND
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm > EPS:
                forward = (blended / blended_norm).astype(np.float32)
                forward[2] = 0.0
        prev_forward = forward.copy()

        base_frame = sequence.frames[plan.base_frame_index]
        actor_data = apply_transform_to_frame(base_frame, sequence, plan.transform)

        npc_vertex_entries: list[np.ndarray] = []
        npc_debug_vertices: list[np.ndarray] = []
        if npc_active and npc_num_frames > 0 and npc_wedge_max_range is not None:
            wedge_forward = forward[:2]
            norm_fwd = float(np.linalg.norm(wedge_forward))
            if norm_fwd < EPS:
                wedge_forward = np.array([0.0, 1.0], dtype=np.float32)
            else:
                wedge_forward = (wedge_forward / norm_fwd).astype(np.float32)
            wedge = CameraWedge(
                origin_xy=camera_position[:2],
                forward_xy=wedge_forward,
                fov_deg=fov_deg,
                max_range=float(npc_wedge_max_range),
                min_range=npc_config.min_distance_from_camera,
            )
            npc_seed = _npc_frame_seed(npc_seed_base, scene_id, label_id, idx)
            rng = np.random.default_rng(npc_seed)
            placement = plan_npc_positions(
                wedge=wedge,
                free_mask=npc_free_mask,
                meta=npc_meta,
                rng=rng,
                config=npc_config,
                goal_xy=npc_goal,
                exclude_discs=[(plan.actor_pos_xy, npc_config.clearance_radius)],
            )
            npc_shortfall_total += placement.shortfall
            orient_rng = np.random.default_rng(npc_seed + 1)
            for npc_i, npc_xy in enumerate(placement.positions_xy):
                yaw = float(orient_rng.uniform(-math.pi, math.pi))
                rotation_np = rotation_matrix_z_np(yaw)
                translation_vec = np.array([npc_xy[0], npc_xy[1], npc_ground_z], dtype=np.float64)
                transform = build_transform_matrix(rotation_np, translation_vec)
                if npc_options.loop:
                    npc_anim_idx = int((idx * npc_anim_step + npc_i) % npc_num_frames)
                else:
                    npc_anim_idx = min(int(idx * npc_anim_step + npc_i), max(npc_num_frames - 1, 0))
                base_npc_frame = npc_sequence.frames[npc_anim_idx]
                npc_data = apply_transform_to_frame(base_npc_frame, npc_sequence, transform)
                npc_vertex_entries.append(ply_utils.align_dtype(npc_data, scene_dtype))
                if dump_enabled and (dump_max is None or dump_count < dump_max) and (idx % dump_stride == 0):
                    npc_debug_vertices.append(_build_actor_debug_vertices(npc_data, npc_sequence))

        if dump_enabled and (dump_max is None or dump_count < dump_max) and (idx % dump_stride == 0):
            actor_vertices = _build_actor_debug_vertices(actor_data, sequence)
            entries: list[np.ndarray] = []
            if scene_debug_vertices is not None and scene_debug_vertices.size > 0:
                entries.append(scene_debug_vertices)
            entries.append(actor_vertices)
            if npc_debug_vertices:
                entries.extend(npc_debug_vertices)
            debug_vertices = entries[0] if len(entries) == 1 else np.concatenate(entries, axis=0)
            dump_path = dump.directory / f"{frame_prefix}_{idx:04d}.ply"
            _write_debug_ply(dump_path, debug_vertices)
            dump_count += 1

        if dump and dump.dump_only:
            continue

        actor_aligned = ply_utils.align_dtype(actor_data, scene_dtype)
        entries_for_concat = [scene_subset, actor_aligned] + npc_vertex_entries
        combined_vertices = ply_utils.concat_vertices(*entries_for_concat)
        scene_template.write(combined_vertices, combined_tmp_path)

        frame_gaussians.load_ply(str(combined_tmp_path))

        target_xy = camera_position[:2] + forward[:2] * look_ahead
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

        img_pkg = render_or(camera, frame_gaussians, pipeline, bg_color=bg_color, orthographic=False)
        render = img_pkg['render'].detach().cpu().numpy()
        render_uint8 = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
        render_uint8 = np.rot90(render_uint8, k=2)

        if video:
            with (
                metrics.measure(PathMetricRecorder.VIDEO_STAGE)
                if metrics is not None
                else contextlib.nullcontext()
            ):
                writer.append_data(render_uint8)
        elif save_rgb_frames:
            frame_path = frames_dir / f"{frame_prefix}_{frame_counter:04d}.png"
            with (
                metrics.measure(PathMetricRecorder.PNG_STAGE)
                if metrics is not None
                else contextlib.nullcontext()
            ):
                imageio.imwrite(frame_path, render_uint8)
        if save_depth_maps:
            try:
                _save_depth_map(
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=frame_counter,
                    img_pkg=img_pkg,
                    orthographic=False,
                    depth_bit_depth=depth_bit_depth,
                    metrics=metrics,
                )
            except Exception as e:
                print(f"[WARN] Failed to save depth for frame {frame_counter}: {e}", flush=True)
        if save_camera_metadata:
            try:
                _save_camera_metadata(
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=frame_counter,
                    camera=camera,
                    orthographic=False,
                )
            except Exception as e:
                print(f"[WARN] Failed to save camera metadata for frame {frame_counter}: {e}", flush=True)
        frame_counter += 1
        if metrics:
            metrics.sample_vram()

        if verbose and (idx % 10 == 0 or idx == total_steps - 1):
            print(
                f"[VERBOSE] Scene {scene_id} / {label_id}: frame {idx + 1}/{total_steps} complete.",
                flush=True,
            )

    if verbose:
        print(
            f"[VERBOSE] Scene {scene_id} / {label_id}: actor rendering complete ({frame_counter} frames).",
            flush=True,
        )
    if npc_active and verbose and npc_shortfall_total > 0:
        print(
            f"[VERBOSE] Scene {scene_id} / {label_id}: NPC placement shortfall total {npc_shortfall_total}.",
            flush=True,
        )

    return camera_xy_seq, actor_xy_seq



def resolve_label_directory(scene_task_dir: Path) -> Path | None:
    """Return the directory that stores label path JSON files for a scene."""

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


def find_ply_file(dataset_dir: Path) -> Path:
    """Return the first plausible .ply inside the dataset directory."""

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


def prepare_path_data(
    json_path: Path,
    meta: dict,
    stride: int,
    mirror_translation: bool,
    swap_xy: bool,
) -> PreparedPath:
    raw_points, raster_pixels = load_raster_world_points(json_path, swap_xy=swap_xy)
    a_x, b_x, a_y, b_y = derive_affine_transform(raw_points, raster_pixels, meta)
    transformed = [
        np.array([a_x * pt[0] + b_x, a_y * pt[1] + b_y], dtype=np.float32)
        for pt in raw_points
    ]

    points_xy = deduplicate_points(transformed)
    sampled_xy = sample_points(points_xy, stride)
    if len(sampled_xy) < 2:
        sampled_xy = points_xy

    if mirror_translation:
        center_x = 0.5 * (meta["left"] + meta["right"])
        center_y = 0.5 * (meta["top"] + meta["bottom"])
        path_xy = [
            np.array([center_x * 2.0 - pt[0], center_y * 2.0 - pt[1]], dtype=np.float32)
            for pt in sampled_xy
        ]
    else:
        path_xy = [np.array([pt[0], pt[1]], dtype=np.float32) for pt in sampled_xy]

    return PreparedPath(
        path_xy=path_xy,
        raw_points=raw_points,
        sampled_xy=sampled_xy,
        affine=(a_x, b_x, a_y, b_y),
        floor_z=meta["lower_z"],
        ceiling=meta["upper_z"],
    )


def estimate_actor_frame_count(path_xy: Sequence[np.ndarray], follow_distance: float) -> int:
    if not path_xy:
        return 0
    if len(path_xy) == 1:
        return 1

    sampler = PathSampler(path_xy)
    distances = list(sampler.cumulative)
    total_length = sampler.total_length
    follow_distance_m = max(float(follow_distance), 0.0)
    max_camera_distance = max(total_length - follow_distance_m, 0.0)

    count = 0
    for dist in distances:
        camera_distance = min(dist, max_camera_distance)
        count += 1
        if camera_distance >= max_camera_distance - 1e-6:
            break
    return count


def build_path_metadata(
    *,
    scene_id: str,
    label_id: str,
    path_xy: Sequence[np.ndarray],
    camera_xy_seq: Sequence[np.ndarray],
    meta: dict,
    follow_distance: float,
    limit_to_follow: bool,
) -> dict:
    """
    Assemble per-frame metadata describing camera/person positions along the path.
    """
    follow_distance_m = max(float(follow_distance), 0.0)
    sampler = PathSampler(path_xy)
    cumulative = sampler.cumulative
    total_length = sampler.total_length
    max_camera_distance = max(total_length - follow_distance_m, 0.0)

    frames: list[dict] = []
    distances: list[float] = []

    for dist in cumulative:
        if len(distances) >= len(camera_xy_seq):
            break
        camera_distance = min(dist, max_camera_distance) if limit_to_follow else min(dist, total_length)
        distances.append(camera_distance)
        if limit_to_follow and camera_distance >= max_camera_distance - 1e-6:
            # Camera stops once we can no longer keep the desired follow distance.
            break

    # Guard against any unexpected mismatch by padding with the last known distance.
    while len(distances) < len(camera_xy_seq):
        distances.append(distances[-1] if distances else 0.0)

    points = sampler.points
    for frame_idx, (camera_xy, cam_dist) in enumerate(zip(camera_xy_seq, distances)):
        person_distance = min(cam_dist + follow_distance_m, total_length)
        person_xy = sampler.position_at(person_distance)

        between_world: list[list[float]] = []
        between_pixel: list[list[int]] = []
        for point_idx, point in enumerate(points):
            dist_val = cumulative[point_idx]
            if cam_dist < dist_val < person_distance - 1e-6:
                bw = [float(point[0]), float(point[1])]
                between_world.append(bw)
                pixel = world_to_pixel(meta, np.array(point[:2], dtype=np.float32))
                between_pixel.append([int(pixel[0]), int(pixel[1])])

        frame_entry = {
            "id": int(frame_idx),
            "camera_world": [float(camera_xy[0]), float(camera_xy[1])],
            "person_world": [float(person_xy[0]), float(person_xy[1])],
            "between_world": between_world,
            "between_pixel": between_pixel,
        }
        frames.append(frame_entry)

    return {
        "scene": scene_id,
        "label": label_id,
        "follow_distance": follow_distance_m,
        "frames": frames,
    }

def get_forward_fn(args):
    return forward_direction_beta if getattr(args, "use_forward_beta", False) else forward_direction

def forward_direction(points: Sequence[np.ndarray], idx: int, window: int = 1) -> np.ndarray:
    """Estimate the forward direction (in XY) around the given index."""

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
    """
    Robust local forward (XY). Symmetric, distance-weighted; friendlier at ends.
    Returns a unit 3D vector [dx, dy, 0].
    """
    n = len(points)
    if n == 1:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    w = max(1, int(window))
    accum = np.zeros(2, dtype=np.float64)
    weight_sum = 0.0

    # symmetric neighborhood; clamp to ends
    for k in range(1, w + 1):
        i0 = max(idx - k, 0)
        i1 = min(idx + k, n - 1)
        # prefer a longer baseline when possible
        if i1 > idx:
            d_fwd = points[i1][:2] - points[idx][:2]
            nf = np.linalg.norm(d_fwd)
            if nf > 1e-6:
                wf = k  # weight grows with span
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
    scene_template: ply_utils.GaussianPly,
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
    default_follow_distance: float,
    verbose: bool,
    hide_actor_enabled: bool = False,
    mirror_bev_x = False,
    mirror_bev_y = False,
    save_rgb_frames: bool = False,
    save_depth_maps: bool = True,
    save_camera_metadata: bool = True,
    depth_bit_depth: int = 16,
    save_follow_metadata: bool = True,
    actor_dump_root: Path | None = None,
    actor_dump_stride: int = 30,
    actor_dump_include_scene: bool = True,
    actor_dump_max_frames: int | None = 25,
    actor_dump_only: bool = False,
    gpu_only: bool = False,
    prepared_path: PreparedPath | None = None,
    navdp_manager: NavdpPlyCoordinator | None = None,
    metrics_enabled: bool = False,
    job_slot: int | None = None,
    job_actor_id: str | None = None,
    job_name: str | None = None,
    npc_render: bool = False,
    npc_config: NPCDensityConfig | None = None,
    npc_free_mask: np.ndarray | None = None,
    npc_meta: dict | None = None,
    npc_seed_base: int = 0,
    npc_max_range: float | None = None,
    npc_forward_fn=None,
    npc_goal_xy: np.ndarray | None = None,
    npc_actor_runtime: ActorRuntime | None = None,
) -> dict | None:
    """Render frames for a single raster_world trajectory."""

    start_time = time.perf_counter()
    frames_rendered = 0
    metrics_recorder = PathMetricRecorder(device=device) if metrics_enabled else None
    if metrics_recorder:
        metrics_recorder.sample_vram()

    path_data = prepared_path or prepare_path_data(
        json_path=json_path,
        meta=meta,
        stride=stride,
        mirror_translation=mirror_translation,
        swap_xy=swap_xy,
    )
    path_xy = path_data.path_xy
    if len(path_xy) < 2:
        raise ValueError(f"Need at least two distinct points in {json_path}")

    floor_z = path_data.floor_z
    ceiling = path_data.ceiling
    camera_z = ceiling + height_offset
    a_x, b_x, a_y, b_y = path_data.affine

    used_follow_distance = (
        float(actor_runtime.options.follow_distance)
        if actor_runtime is not None
        else float(default_follow_distance)
    )

    positions = [
        np.array([xy[0], xy[1], camera_z], dtype=np.float32) for xy in path_xy
    ]
    npc_enabled = (
        npc_render
        and npc_config is not None
        and npc_free_mask is not None
        and npc_meta is not None
        and npc_actor_runtime is not None
    )
    npc_goal = npc_goal_xy if npc_goal_xy is not None else (path_xy[-1] if path_xy else None)
    npc_wedge_max_range = None
    npc_forward_fn = npc_forward_fn if npc_forward_fn is not None else forward_direction
    if npc_enabled:
        npc_wedge_max_range = (
            npc_max_range
            if npc_max_range is not None
            else (npc_config.max_distance_from_camera if npc_config.max_distance_from_camera is not None else 5.0)
        )
        npc_wedge_max_range = float(npc_wedge_max_range)

    scene_vertices: np.ndarray | None = None
    scene_dtype: np.dtype | None = None
    combined_tmp_dir: Path | None = None
    if not gpu_only:
        scene_vertices = np.array(scene_template.data, copy=False)
        scene_dtype = scene_template.data.dtype
        combined_tmp_dir = output_dir / scene_id / "__tmp_actor_combined"

    dump_options: ActorDumpOptions | None = None
    if actor_dump_only and actor_runtime is None:
        raise ValueError("--actor-dump-only requires an animated actor sequence.")
    if actor_dump_only and hide_actor_enabled:
        raise ValueError("--actor-dump-only is incompatible with --hide-actor.")
    if actor_dump_root is not None:
        label_dump_dir = actor_dump_root / scene_id / json_path.stem
        max_frames_opt: int | None = None
        if actor_dump_max_frames is not None and actor_dump_max_frames > 0:
            max_frames_opt = actor_dump_max_frames
        dump_options = ActorDumpOptions(
            directory=label_dump_dir,
            stride=max(1, actor_dump_stride),
            include_scene=bool(actor_dump_include_scene),
            max_frames=max_frames_opt,
            dump_only=bool(actor_dump_only),
        )
    elif actor_dump_only:
        raise ValueError("--actor-dump-only requires --actor-dump-ply-dir to be set.")

    effective_video = video
    if dump_options is not None and dump_options.dump_only:
        effective_video = False

    if debug:
        print(f"[DEBUG] Processing label config: {json_path}", flush=True)
        print(
            f"[DEBUG] Affine transform: x' = {a_x:.6f} * x + {b_x:.6f}, "
            f"y' = {a_y:.6f} * y + {b_y:.6f} (swap_xy={swap_xy})",
            flush=True,
        )
        raw_preview = [tuple(map(float, pt)) for pt in path_data.raw_points[:5]]
        preview = [tuple(map(float, pts)) for pts in path_data.sampled_xy[:5]]
        print(
            f"[DEBUG] Raw points: {raw_preview} -> transformed: {preview}",
            flush=True,
        )
        if mirror_translation:
            center_x = 0.5 * (meta["left"] + meta["right"])
            center_y = 0.5 * (meta["top"] + meta["bottom"])
            mirrored_preview = [
                (
                    float(2.0 * center_x - pt[0]),
                    float(2.0 * center_y - pt[1]),
                )
                for pt in path_data.sampled_xy[:5]
            ]
            print(
                f"[DEBUG] Mirrored (translation-adjusted) XY: {mirrored_preview}",
                flush=True,
            )

    frames_dir = output_dir / scene_id / json_path.stem
    video_dir = output_dir / scene_id
    video_path = video_dir / f"{json_path.stem}.mp4"
    
    # Always create frames_dir since per-frame outputs (camera/depth/RGB) live there.
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    if effective_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite and video_path.exists():
            print(f"  Skipping {json_path.stem}: video already exists.")
            return
    
    # Check for existing frames if not in video mode
    if not effective_video:
        skip_frame_check = dump_options is not None and dump_options.dump_only
        if not overwrite and not skip_frame_check:
            existing = list(frames_dir.glob("frame_*.png"))
            if existing:
                print(f"  Skipping {json_path.stem}: frames already exist ({len(existing)} files).")
                return

    with torch.no_grad():
        prev_forward: np.ndarray | None = None
        direction_window = STABILIZE_WINDOW if stabilize else 1
        cam_seq: list[np.ndarray] = []
        act_seq: list[np.ndarray] | None = None
        writer_ctx = (
            imageio.get_writer(
                video_path,
                mode="I",
                fps=DEFAULT_VIDEO_FPS,
            )
            if effective_video
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
                    cam_seq, act_seq = render_actor_camera_only_sequence(
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
                        video=effective_video,
                        save_rgb_frames=save_rgb_frames,
                        save_depth_maps=save_depth_maps,
                        save_camera_metadata=save_camera_metadata,
                        depth_bit_depth=depth_bit_depth,
                        frames_dir=frames_dir,
                        frame_prefix="frame",
                        debug=debug,
                        stabilize=stabilize,
                        verbose=verbose,
                        dump=dump_options,
                        metrics=metrics_recorder,
                    )
                    frames_rendered = len(cam_seq)
                else:
                    cam_seq, act_seq = render_actor_follow_sequence(
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
                        video=effective_video,
                        save_rgb_frames=save_rgb_frames,
                        save_depth_maps=save_depth_maps,
                        save_camera_metadata=save_camera_metadata,
                        depth_bit_depth=depth_bit_depth,
                        frames_dir=frames_dir,
                        frame_prefix="frame",
                        debug=debug,
                        stabilize=stabilize,
                        verbose=verbose,
                        dump=dump_options,
                        scene_vertices=scene_vertices,
                        scene_dtype=scene_dtype,
                        scene_template=scene_template,
                        combined_tmp_dir=combined_tmp_dir,
                        sh_degree=gaussians.max_sh_degree,
                        gpu_only=gpu_only,
                        metrics=metrics_recorder,
                        npc_enabled=npc_enabled,
                        npc_config=npc_config,
                        npc_free_mask=npc_free_mask,
                        npc_meta=npc_meta,
                        npc_seed_base=npc_seed_base,
                        npc_wedge_max_range=npc_wedge_max_range,
                        npc_forward_fn=npc_forward_fn,
                        npc_goal=npc_goal,
                        npc_actor_runtime=npc_actor_runtime,
                    )
                    frames_rendered = len(cam_seq)
                if render_bev:
                    if verbose:
                        print(
                            f"[VERBOSE] Scene {scene_id} / {json_path.stem}: saving BEV debug image.",
                            flush=True,
                        )
                    # Decide where to save: same folder as the video
                    bev_dir = output_dir / scene_id  # same as video_dir
                    bev_path = bev_dir / f"{json_path.stem}_BEV.png"

                    # Build sequences for BEV depending on branch
                    if actor_runtime is not None:
                        if hide_actor_enabled:
                            # camera-only per-point (actor overlay hidden)
                            actor_path_dotted = True
                        else:
                            actor_path_dotted = False
                    else:
                        # non-actor branch: we used 'positions' for camera
                        cam_seq = [p[:2].copy() for p in positions]
                        act_seq = None
                        actor_path_dotted = False

                    # Total length estimate from sampler over path_xy
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
                cam_seq = [p[:2].copy() for p in positions]
                act_seq = None
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

                    if save_rgb_frames:
                        try:
                            frame_path = frames_dir / f"frame_{idx:04d}.png"
                            timing_ctx = (
                                metrics_recorder.measure(PathMetricRecorder.PNG_STAGE)
                                if metrics_recorder is not None
                                else contextlib.nullcontext()
                            )
                            with timing_ctx:
                                imageio.imwrite(frame_path, render_uint8)
                        except Exception as e:
                            print(f"[WARN] Failed to save RGB frame {idx}: {e}", flush=True)
                    if save_depth_maps:
                        try:
                            _save_depth_map(
                                frames_dir=frames_dir,
                                frame_prefix="frame",
                                frame_idx=idx,
                                img_pkg=img_pkg,
                                orthographic=orthographic,
                                depth_bit_depth=depth_bit_depth,
                                metrics=metrics_recorder,
                            )
                        except Exception as e:
                            print(f"[WARN] Failed to save depth for frame {idx}: {e}", flush=True)
                    if save_camera_metadata:
                        try:
                            _save_camera_metadata(
                                frames_dir=frames_dir,
                                frame_prefix="frame",
                                frame_idx=idx,
                                camera=camera,
                                orthographic=orthographic,
                            )
                        except Exception as e:
                            print(f"[WARN] Failed to save camera metadata for frame {idx}: {e}", flush=True)
                    
                    # Also write to video if requested
                    if video:
                        try:
                            timing_ctx = (
                                metrics_recorder.measure(PathMetricRecorder.VIDEO_STAGE)
                                if metrics_recorder is not None
                                else contextlib.nullcontext()
                            )
                            with timing_ctx:
                                writer.append_data(render_uint8)
                        except Exception as e:
                            print(f"[ERROR] Failed to append frame {idx} to video: {e}", flush=True)
                            raise

                    if metrics_recorder is not None:
                        metrics_recorder.sample_vram()

                    if verbose and (idx % 10 == 0 or idx == total_positions - 1):
                        print(
                            f"[VERBOSE] Scene {scene_id} / {json_path.stem}: frame {idx + 1}/{total_positions} complete.",
                            flush=True,
                        )
                frames_rendered = total_positions

            if verbose:
                print(
                    f"[VERBOSE] Scene {scene_id} / {json_path.stem}: render finished.",
                    flush=True,
                )

    duration = time.perf_counter() - start_time
    if frames_rendered > 0:
        print(
            f"      Timing: {duration:.2f}s total, {duration / frames_rendered:.3f}s per frame ({frames_rendered} frames).",
            flush=True,
        )
    else:
        print(
            f"      Timing: {duration:.2f}s total (no frames rendered).",
            flush=True,
        )

    if save_follow_metadata:
        metadata_payload = build_path_metadata(
            scene_id=scene_id,
            label_id=json_path.stem,
            path_xy=path_xy,
            camera_xy_seq=cam_seq,
            meta=meta,
            follow_distance=used_follow_distance,
            limit_to_follow=actor_runtime is not None,
        )
        metadata_path = output_dir / scene_id / f"{json_path.stem}_follow_path.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as meta_fh:
            json.dump(metadata_payload, meta_fh, indent=2)

    if navdp_manager is not None:
        try:
            timing_ctx = (
                metrics_recorder.measure(PathMetricRecorder.PLY_STAGE)
                if metrics_recorder is not None
                else contextlib.nullcontext()
            )
            with timing_ctx:
                navdp_manager.add_path(
                    scene_id=scene_id,
                    label_id=json_path.stem,
                    path_xy=path_xy,
                    meta=meta,
                    gaussians=gaussians,
                    output_dir=output_dir,
                )
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"      WARNING: Failed to build NavDP mask PLY: {exc}",
                flush=True,
            )

    summary = None
    if metrics_recorder is not None:
        summary = metrics_recorder.finalize(
            scene_id=scene_id,
            label_id=json_path.stem,
            frames_rendered=frames_rendered,
            total_duration=duration,
            video_enabled=bool(effective_video),
            job_slot=job_slot,
            job_actor_id=job_actor_id,
            job_name=job_name,
        )
    return summary


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Render frames along raster_world navigation paths.")
    parser.add_argument(
        "--offload-nas-dir",
        type=Path,
        default=None,
        help="If set, when local free space drops below threshold, move finished path outputs to this NAS mirror (e.g. /mnt/nas/jiankundong/path_video_frames_Jiankun_test)."
    )
    parser.add_argument(
        "--offload-min-free-gb",
        type=float,
        default=0.5,
        help="Minimum free space (in GB) to maintain locally before offloading (default: 0.5)."
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR,
        help=f"Scene reconstruction root (default: {SCENES_DIR}).",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=TASK_OUTPUT_DIR,
        help=f"Root containing per-scene label JSON files (default: {TASK_OUTPUT_DIR}).",
    )

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
        default=0, #-0.098 for LHM model following scene generations
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
        "--resume-from",
        type=str,
        default=None,
        help="Skip scenes until reaching the first whose numeric prefix matches this value (e.g. '0893').",
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
        help="Print debug info such as currently loaded PLY file paths.",
    )
    parser.add_argument(
        "--swap-xy",
        action="store_true",
        help="Swap x/y when reading raster_world coordinates.",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        default=True,
        help="Smooth the forward view direction to reduce camera shake.",
    )
    parser.add_argument(
        "--video",
        action=BooleanOptionalAction,
        default=True,
        help="Write composited MP4 video (default). Use --no-video to keep per-frame PNGs only.",
    )
    parser.add_argument(
        "--rgb-frames",
        action=BooleanOptionalAction,
        default=False,
        help="Save RGB PNGs for each frame; use --no-rgb-frames to disable if only depth/camera logs are needed.",
    )
    parser.add_argument(
        "--save-depth-maps",
        action=BooleanOptionalAction,
        default=True,
        help="Persist per-frame depth PNGs (default: on).",
    )
    parser.add_argument(
        "--save-camera-metadata",
        action=BooleanOptionalAction,
        default=True,
        help="Write per-frame camera metadata JSON files (default: on).",
    )
    parser.add_argument(
        "--depth-bit-depth",
        type=int,
        choices=sorted(DEPTH_ENCODING_SPECS.keys()),
        default=16,
        help=(
            "Depth PNG quantization: 16-bit=1mm to 65.535m, 12-bit=1mm to 4.095m, "
            "10-bit=2mm to 2.046m, 8-bit=4cm to 10.2m."
        ),
    )
    parser.add_argument(
        "--save-follow-metadata",
        action=BooleanOptionalAction,
        default=True,
        help="Write per-path follow metadata JSON (separate from per-frame camera metadata; default: on).",
    )
    parser.add_argument(
        "--no-mirror-translation",
        action="store_true",
        help="Disable the default reflection correction that mirrors XY relative to the scene center.",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERROR_LOG,
        help=f"Append rendering failures to this log file (default: {DEFAULT_ERROR_LOG}).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="If set, write per-path runtime metrics to this JSON file for external progress tracking.",
    )
    parser.add_argument(
        "--job-slot",
        type=int,
        default=None,
        help="Optional worker slot identifier used when aggregating metrics across threads.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Friendly job name for metrics reporting (defaults to scene/actor pairing).",
    )
    parser.add_argument(
        "--job-actor-id",
        type=str,
        default=None,
        help="Actor identifier for metrics reporting when invoked via the parallel runner.",
    )
    parser.add_argument(
        "--verbose",
        action=BooleanOptionalAction,
        default=False,
        help="Print detailed rendering progress information.",
    )
    parser.add_argument(
        "--hide-actor",
        action="store_true",
        default=False,
        help="Disable animated actor overlay rendering."
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
        default=1.5,
        help="Desired distance (meters) the camera trails behind the actor along the path (default: 1.5).",
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
        "--actor-dump-ply-dir",
        type=Path,
        default=None,
        help="Directory to store debug actor/scene PLY dumps.",
    )
    parser.add_argument(
        "--actor-dump-stride",
        type=int,
        default=30,
        help="Write a debug PLY every Nth actor frame when dumping is enabled (default: 30).",
    )
    parser.add_argument(
        "--actor-dump-max",
        type=int,
        default=20,
        help="Maximum number of debug PLYs per label (default: 20). Use 0 for unlimited.",
    )
    parser.add_argument(
        "--actor-dump-include-scene",
        action=BooleanOptionalAction,
        default=True,
        help="Include the cropped scene gaussians when dumping actor PLYs (default: on).",
    )
    parser.add_argument(
        "--actor-dump-only",
        action="store_true",
        default=False,
        help="Skip rasterization and only generate actor debug PLYs (requires --actor-dump-ply-dir).",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        default=False,
        help="Compose scene and actor entirely on GPU (no temporary PLYs, higher VRAM usage).",
    )
    parser.add_argument(
        "--animation-cycle-mod",
        type=int,
        default=3,
        help="Speed modifier for looping actor animation cycles (default: 3 for 30fps -> 10fps).",
    )
    parser.add_argument(
        "--minimal-frames",
        type=int,
        default=None,
        help="Skip paths whose rendered frame count (after follow distance) would fall below this threshold.",
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
        default=True,
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
    parser.add_argument(
        "--navdp-mask-ply",
        action="store_true",
        default=False,
        help="Enable NavDP occupancy-masked PLY exports (default: off).",
    )
    parser.add_argument(
        "--navdp-ply-per-scene",
        action="store_true",
        default=False,
        help="Combine all paths from a scene into a single NavDP PLY instead of one per path (requires --navdp-mask-ply).",
    )
    parser.add_argument(
        "--npc-bev-debug",
        action="store_true",
        default=False,
        help="Generate NPC placement BEV debug images alongside rendering.",
    )
    parser.add_argument(
        "--npc-bev-debug-only",
        action="store_true",
        default=False,
        help="Only generate NPC placement BEV debug images (skip rendering).",
    )
    parser.add_argument(
        "--npc-render",
        action="store_true",
        default=False,
        help="Render NPCs into RGB/depth using the density planner (not just BEV overlays).",
    )
    parser.add_argument(
        "--npc-actor-dir",
        type=Path,
        default=None,
        help="Optional actor sequence directory for NPCs (defaults to --actor-seq-dir or first child of --npc-actor-root).",
    )
    parser.add_argument(
        "--npc-bev-output-dir",
        type=Path,
        default=None,
        help="Output root for NPC BEV debug images (defaults to --output-dir).",
    )
    parser.add_argument(
        "--npc-density-coverage",
        type=float,
        default=None,
        help="Target NPC area coverage fraction (0..1) within the camera wedge.",
    )
    parser.add_argument(
        "--npc-max-count",
        type=int,
        default=None,
        help="Hard cap on NPCs per frame (applied after coverage target).",
    )
    parser.add_argument(
        "--npc-count",
        type=int,
        default=None,
        help="Desired NPC count per frame (soft unless npc-priority=count).",
    )
    parser.add_argument(
        "--npc-priority",
        type=str,
        choices=["coverage", "count"],
        default="coverage",
        help="Treat coverage or count as the hard requirement (coverage still acts as a cap).",
    )
    parser.add_argument(
        "--npc-density-mode",
        type=str,
        choices=["area", "angular"],
        default="angular",
        help="Coverage metric: 'area' uses wedge area; 'angular' uses FOV angular span (default: angular).",
    )
    parser.add_argument(
        "--npc-zone-ratio",
        type=str,
        default="1:2:1",
        help="Near:mid:far ratio for distributing NPCs radially (default: 1.0:2.0:1.0; applied when target count >= 12).",
    )
    parser.add_argument(
        "--npc-clearance",
        type=float,
        default=0.30,
        help="NPC collision radius in meters (default: 0.30).",
    )
    parser.add_argument(
        "--npc-min-distance",
        type=float,
        default=1.0,
        help="Minimum radial distance from the camera when placing NPCs (default: 1.0 m).",
    )
    parser.add_argument(
        "--npc-max-range",
        type=float,
        default=5.0,
        help="Maximum radial distance from the camera when placing NPCs (default: 5.0 m).",
    )
    parser.add_argument(
        "--npc-allow-blocking",
        action=BooleanOptionalAction,
        default=False,
        help="Allow NPCs to block the camera->goal line of sight (default: off).",
    )
    parser.add_argument(
        "--npc-free-threshold",
        type=int,
        default=250,
        help="Occupancy pixel threshold treated as free space (default: 250 => near-white only).",
    )
    parser.add_argument(
        "--npc-free-white",
        action=BooleanOptionalAction,
        default=True,
        help="Interpret free space as white pixels >= threshold (default: on). Disable to treat dark pixels <= threshold as free.",
    )
    parser.add_argument(
        "--npc-auto-clearance",
        action=BooleanOptionalAction,
        default=False,
        help="Derive NPC clearance from HumanGS sources (trimmed mean radius) and override the default clearance if larger.",
    )
    parser.add_argument(
        "--npc-actor-root",
        type=Path,
        default=BASE_DIR / "data" / "human_gs_source",
        help="Root directory of HumanGS actor sources for auto clearance computation.",
    )
    parser.add_argument(
        "--npc-radius-samples",
        type=int,
        default=80,
        help="Maximum number of actors to sample when auto-computing clearance (default: 80).",
    )
    parser.add_argument(
        "--npc-max-resamples",
        type=int,
        default=50,
        help="Maximum resampling attempts per NPC placement (default: 50).",
    )
    parser.add_argument(
        "--npc-seed",
        type=int,
        default=12345,
        help="Base seed for NPC placement determinism.",
    )

    parser.set_defaults(actor_loop=True)
    return parser


def normalise_label_ids(raw_ids: list[str] | None) -> set[str]:
    """Convert label identifiers to their stem form for easy comparison."""

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
    offloader = _OffloadWorker(verbose=True)
    parser = parse_args()
    args = parser.parse_args()
    # Allow overriding dataset roots via CLI to keep worker commands consistent with task planning.
    global SCENES_DIR, TASK_OUTPUT_DIR  # noqa: PLW0603
    SCENES_DIR = args.scenes_dir
    TASK_OUTPUT_DIR = args.tasks_dir

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for rendering but not available.")
    device = torch.device("cuda")

    label_filter = normalise_label_ids(args.label_ids)
    width, height = args.resolution
    nas_offload_dir: Path | None = args.offload_nas_dir
    offload_min_free_bytes: int = int(float(args.offload_min_free_gb) * (1024**3))
    ensure_output_dir_writable(args.output_dir)
    print(f"[CHECK] Output directory ready: {args.output_dir}", flush=True)

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
    save_rgb_frames = bool(args.rgb_frames)
    save_depth_maps = bool(args.save_depth_maps)
    save_camera_metadata = bool(args.save_camera_metadata)
    depth_bit_depth = int(args.depth_bit_depth)
    save_follow_metadata = bool(args.save_follow_metadata)
    npc_render_enabled = bool(args.npc_render)
    actor_runtime: ActorRuntime | None = None
    npc_actor_runtime: ActorRuntime | None = None
    gpu_only_enabled = bool(args.gpu_only)
    actor_dump_root = args.actor_dump_ply_dir
    actor_dump_stride = max(1, int(args.actor_dump_stride))
    actor_dump_include_scene = bool(args.actor_dump_include_scene)
    actor_dump_max = None if args.actor_dump_max == 0 else (int(args.actor_dump_max) if args.actor_dump_max is not None else None)
    actor_dump_only = bool(args.actor_dump_only)
    minimal_frames_required = (
        int(args.minimal_frames)
        if args.minimal_frames is not None and int(args.minimal_frames) > 0
        else None
    )
    navdp_manager = (
        NavdpPlyCoordinator(per_scene=bool(args.navdp_ply_per_scene))
        if args.navdp_mask_ply
        else None
    )
    if args.navdp_ply_per_scene and navdp_manager is None:
        print("  [WARN] --navdp-ply-per-scene ignored without --navdp-mask-ply.", flush=True)
    metrics_enabled = bool(args.metrics_json)
    collected_metrics: list[dict] = []
    npc_bev_enabled = bool(args.npc_bev_debug or args.npc_bev_debug_only)
    npc_bev_only = bool(args.npc_bev_debug_only)
    npc_bev_output_root: Path = args.npc_bev_output_dir if args.npc_bev_output_dir is not None else args.output_dir
    try:
        zone_ratio = _parse_zone_ratio(str(args.npc_zone_ratio))
    except ValueError as exc:
        raise SystemExit(f"Invalid --npc-zone-ratio: {exc}") from exc
    npc_density_config = NPCDensityConfig(
        clearance_radius=float(args.npc_clearance),
        min_distance_from_camera=float(args.npc_min_distance),
        max_distance_from_camera=float(args.npc_max_range) if args.npc_max_range is not None and float(args.npc_max_range) > 0 else None,
        target_coverage=float(args.npc_density_coverage) if args.npc_density_coverage is not None else None,
        max_npcs=int(args.npc_max_count) if args.npc_max_count is not None and int(args.npc_max_count) > 0 else None,
        allow_blocking=bool(args.npc_allow_blocking),
        max_resamples=max(1, int(args.npc_max_resamples)),
        free_pixel_min=int(args.npc_free_threshold),
        free_is_white=bool(args.npc_free_white),
        coverage_mode=str(args.npc_density_mode),
        desired_count=int(args.npc_count) if args.npc_count is not None and int(args.npc_count) > 0 else None,
        priority=str(args.npc_priority),
        zone_weights=zone_ratio,
    )
    if args.npc_auto_clearance:
        try:
            auto_radius = _compute_default_npc_clearance(
                actor_root=args.npc_actor_root,
                sample_limit=max(1, int(args.npc_radius_samples)),
            )
            if auto_radius > npc_density_config.clearance_radius:
                npc_density_config.clearance_radius = auto_radius
            print(f"  [NPC] Auto clearance radius: {auto_radius:.3f} m (using HumanGS sources)", flush=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [WARN] Failed to auto-compute NPC clearance; using {npc_density_config.clearance_radius:.3f} m. Reason: {exc}", flush=True)
    npc_forward_fn = get_forward_fn(args)

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
            animation_cycle_mod=int(args.animation_cycle_mod),
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

    if npc_render_enabled:
        npc_actor_dir: Path | None = args.npc_actor_dir
        if npc_actor_dir is None:
            npc_actor_dir = args.actor_seq_dir or _first_actor_subdir(args.npc_actor_root)
        if npc_actor_dir is None:
            print(
                "[WARN] NPC rendering requested but no actor sequence found (set --npc-actor-dir or --actor-seq-dir); disabling NPCs.",
                flush=True,
            )
            npc_render_enabled = False
        elif actor_runtime is not None and npc_actor_dir.resolve() == actor_runtime.options.sequence_dir.resolve():
            npc_actor_runtime = actor_runtime
            if verbose_enabled:
                print(
                    f"[VERBOSE] NPC rendering will reuse actor sequence: {npc_actor_dir}",
                    flush=True,
                )
        else:
            npc_actor_options = ActorOptions(
                sequence_dir=npc_actor_dir,
                pattern=args.actor_pattern,
                height=float(args.actor_height),
                follow_distance=float(args.follow_distance),
                buffer_distance=float(args.follow_buffer),
                speed=float(args.actor_speed),
                fps=float(args.actor_fps),
                loop=bool(args.actor_loop),
                foot_offset=float(args.actor_foot_offset),
                animation_cycle_mod=int(args.animation_cycle_mod),
            )
            if npc_actor_options.fps <= 0.0:
                raise ValueError("NPC actor FPS must be positive.")
            if npc_actor_options.speed <= 0.0:
                raise ValueError("NPC actor walking speed must be positive.")
            if npc_actor_options.buffer_distance > npc_actor_options.follow_distance:
                raise ValueError("NPC follow buffer must be less than or equal to follow distance.")
            npc_actor_sequence = load_actor_sequence(
                npc_actor_options,
                debug=debug_enabled,
            )
            npc_actor_runtime = ActorRuntime(options=npc_actor_options, sequence=npc_actor_sequence)
            if verbose_enabled:
                print(
                    f"[VERBOSE] NPC actor sequence ready: {len(npc_actor_sequence.frames)} frames from {npc_actor_dir}.",
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
        # TEST remove this later
        # scene_ids = scene_ids[2:]
        # remove above
        if not scene_ids:
            print(
                f"WARNING: No scenes found under {TASK_OUTPUT_DIR}; exiting.",
                flush=True,
            )
            return
        print(
            f"No --scene provided; traversing all {len(scene_ids)} scenes with label data.",
            flush=True,
        )

    resume_token = args.resume_from.strip() if args.resume_from else None
    if resume_token:
        filtered_ids: list[str] = []
        resume_found = False
        for scene_id in scene_ids:
            if not resume_found and _scene_prefix(scene_id) == resume_token:
                resume_found = True
            if resume_found:
                filtered_ids.append(scene_id)
        if filtered_ids:
            skipped = len(scene_ids) - len(filtered_ids)
            scene_ids = filtered_ids
            print(
                f"Resume token {resume_token} matched; skipped {skipped} scenes, {len(scene_ids)} remaining.",
                flush=True,
            )
        else:
            print(
                f"WARNING: No scenes found with prefix {resume_token}; processing all scenes.",
                flush=True,
            )

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
            print(f"[{scene_idx}/{total_scenes}] Processing scene {scene_id}", flush=True)
            dataset_dir = SCENES_DIR / scene_id
            scene_task_root = TASK_OUTPUT_DIR / scene_id
            label_dir = resolve_label_directory(scene_task_root)

            if not dataset_dir.is_dir():
                print(f"WARNING: Scene directory missing: {dataset_dir}; skipping.")
                continue
            if label_dir is None:
                print(f"WARNING: No label JSON found under {scene_task_root}; skipping.")
                continue

            try:
                meta = load_occupancy_metadata(dataset_dir)
                ply_path = find_ply_file(dataset_dir)
            except FileNotFoundError as exc:
                print(
                    f"WARNING: Scene {scene_id} missing required files ({exc}); skipping to avoid blocking.",
                    flush=True,
                )
                continue
            processed_scenes += 1
            npc_free_mask = None
            npc_occ_image = None
            if npc_bev_enabled or npc_render_enabled:
                try:
                    npc_free_mask, _ = load_free_space_mask(
                        dataset_dir,
                        threshold=int(args.npc_free_threshold),
                        free_is_white=bool(args.npc_free_white),
                    )
                    if npc_bev_enabled:
                        occ_img = imageio.imread(dataset_dir / "occupancy.png")
                        npc_occ_image = _ensure_rgb(np.array(occ_img))
                    print(f"  [NPC] Free-space mask ready (threshold={args.npc_free_threshold}).", flush=True)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"  [WARN] NPC features disabled for {scene_id}: {exc}", flush=True)
                    npc_free_mask = None
                    npc_occ_image = None

            if debug_enabled:
                if label_dir == scene_task_root:
                    print(
                        f"[DEBUG] Using {scene_task_root} as the label configuration directory.",
                        flush=True,
                    )
                print(f"[DEBUG] Loading Gaussian model from: {ply_path}", flush=True)
            print(f"  Loading point cloud: {ply_path.name}", flush=True)
            gaussians = GaussianModel(sh_degree=3)
            gaussians.load_ply(str(ply_path))
            scene_template = ply_utils.GaussianPly.read(ply_path)
            if verbose_enabled:
                base_points = int(gaussians.get_xyz.shape[0])
                _log_vram_usage(
                    f"Scene {scene_id}: loaded base Gaussian model ({base_points} gaussians)",
                    device,
                )

            json_files = sorted(p for p in label_dir.glob("*.json") if not p.name.endswith("_detailed.json"))
            if label_filter:
                json_files = [path for path in json_files if path.stem in label_filter]
            if not json_files:
                print("  No label JSON files found for this scene; skipping.", flush=True)
                continue

            if args.max_labels is not None:
                json_files = json_files[: args.max_labels]

            total_paths_found = len(json_files)
            render_plan: list[tuple[Path, PreparedPath, int]] = []
            skipped_short: list[tuple[Path, int]] = []
            mirror_translation_flag = not args.no_mirror_translation

            for json_path in json_files:
                prepared = prepare_path_data(
                    json_path=json_path,
                    meta=meta,
                    stride=max(1, args.stride),
                    mirror_translation=mirror_translation_flag,
                    swap_xy=swap_xy_enabled,
                )

                est_frames = len(prepared.path_xy)
                if (
                    minimal_frames_required is not None
                    and actor_runtime is not None
                    and not hide_actor_enabled
                ):
                    est_frames = estimate_actor_frame_count(
                        prepared.path_xy,
                        actor_runtime.options.follow_distance,
                    )
                    if est_frames < minimal_frames_required:
                        skipped_short.append((json_path, est_frames))
                        continue

                render_plan.append((json_path, prepared, est_frames))

            total_labels = len(render_plan)
            overall_labels += total_labels

            if skipped_short:
                for skipped_path, est in skipped_short:
                    print(
                        f"    [skip] {skipped_path.stem}: {est} frames < minimal {minimal_frames_required}",
                        flush=True,
                    )

            print(
                f"  {total_labels} label(s) queued for rendering.",
                flush=True,
            )

            for label_idx, (json_path, prepared, _) in enumerate(render_plan, start=1):
                print(
                    f"    [{label_idx:03d}/{total_labels:03d}/{total_paths_found:03d}] -> {json_path.stem}",
                    flush=True,
                )
                if npc_bev_enabled and npc_free_mask is not None and npc_occ_image is not None:
                    npc_summary = generate_npc_bev_debug_for_path(
                        scene_id=scene_id,
                        label_id=json_path.stem,
                        path_xy=prepared.path_xy,
                        meta=meta,
                        occ_image=npc_occ_image,
                        free_mask=npc_free_mask,
                        config=npc_density_config,
                        fov_deg=float(args.fov_deg),
                        seed_base=int(args.npc_seed),
                        output_root=npc_bev_output_root,
                        stabilize=stabilize_enabled,
                        forward_fn=npc_forward_fn,
                        goal_xy=prepared.path_xy[-1] if prepared.path_xy else None,
                        npc_max_range=float(args.npc_max_range) if args.npc_max_range is not None else None,
                        mirror_bev_x=args.bev_mirror_x,
                        mirror_bev_y=args.bev_mirror_y,
                    )
                    if verbose_enabled:
                        print(
                            f"[VERBOSE] NPC BEV debug ({npc_summary['frames']} frames, shortfall {npc_summary['shortfall']}) -> {npc_summary['output_dir']}",
                            flush=True,
                        )
                    else:
                        print(
                            f"      [NPC-BEV] {json_path.stem}: wrote {npc_summary['frames']} frame(s) -> {npc_summary['output_dir']} "
                            f"(shortfall {npc_summary['shortfall']}, coverage mean/min/max "
                            f"{npc_summary['coverage_mean']:.2f}/{npc_summary['coverage_min']:.2f}/{npc_summary['coverage_max']:.2f})",
                            flush=True,
                        )
                if npc_bev_only:
                    continue
                try:
                    summary = render_path_frames(
                        scene_id=scene_id,
                        json_path=json_path,
                        gaussians=gaussians,
                        scene_template=scene_template,
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
                        save_rgb_frames=save_rgb_frames,
                        save_depth_maps=save_depth_maps,
                        save_camera_metadata=save_camera_metadata,
                        depth_bit_depth=depth_bit_depth,
                        save_follow_metadata=save_follow_metadata,
                        debug=debug_enabled,
                        mirror_translation=not args.no_mirror_translation,
                        actor_runtime=actor_runtime,
                        default_follow_distance=float(args.follow_distance),
                        verbose=verbose_enabled,
                        hide_actor_enabled = hide_actor_enabled,
                        render_bev=args.show_BEV,
                        mirror_bev_x=args.bev_mirror_x,
                        mirror_bev_y=args.bev_mirror_y,
                        actor_dump_root=actor_dump_root,
                        actor_dump_stride=actor_dump_stride,
                        actor_dump_include_scene=actor_dump_include_scene,
                        actor_dump_max_frames=actor_dump_max,
                        actor_dump_only=actor_dump_only,
                        gpu_only=gpu_only_enabled,
                        prepared_path=prepared,
                        navdp_manager=navdp_manager,
                        metrics_enabled=metrics_enabled,
                        job_slot=args.job_slot,
                        job_actor_id=args.job_actor_id,
                        job_name=args.job_name,
                        npc_render=npc_render_enabled,
                        npc_config=npc_density_config,
                        npc_free_mask=npc_free_mask,
                        npc_meta=meta,
                        npc_seed_base=int(args.npc_seed),
                        npc_max_range=float(args.npc_max_range) if args.npc_max_range is not None else None,
                        npc_forward_fn=npc_forward_fn,
                        npc_goal_xy=prepared.path_xy[-1] if prepared.path_xy else None,
                        npc_actor_runtime=npc_actor_runtime,
                    )
                    if summary is not None:
                        collected_metrics.append(summary)
                    if nas_offload_dir is not None:
                        # Use the script's local output root as the place to check/move from.
                        maybe_offload_if_low_space(
                            check_path=args.output_dir,
                            min_free_bytes=offload_min_free_bytes,
                            local_out_root=args.output_dir,
                            nas_out_root=nas_offload_dir,
                            scene_id=scene_id,
                            label_stem=json_path.stem,
                            verbose=verbose_enabled,
                            offloader=offloader,
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"      WARNING: Rendering {json_path.name} failed: {exc}", flush=True)
                    log_file = ensure_log_file()
                    error_count += 1
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    log_file.write(
                        f"[{timestamp}] Scene={scene_id} Label={json_path.name} Error={exc}\n"
                    )
                    log_file.write(traceback.format_exc())
                    log_file.write("\n")
                    log_file.flush()

            if not npc_bev_only and navdp_manager is not None:
                try:
                    navdp_manager.finalize_scene(
                        scene_id=scene_id,
                        meta=meta,
                        gaussians=gaussians,
                        output_dir=args.output_dir,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    print(
                        f"      WARNING: Failed to finalize NavDP PLYs for {scene_id}: {exc}",
                        flush=True,
                    )

            print(f"  Finished scene {scene_id}\n", flush=True)
            if nas_offload_dir is not None:
                if verbose_enabled:
                    print(
                        "[CONSOLIDATE] Final pass: moving any remaining outputs to NAS ...",
                        flush=True,
                    )
                offloader.enqueue(consolidate_outputs_to_nas,
                                  local_out_root=args.output_dir,
                                  nas_out_root=nas_offload_dir,
                                  verbose=verbose_enabled)
        offloader.flush_and_stop()
    finally:
        if error_log_file is not None:
            error_log_file.close()

    if args.metrics_json is not None:
        metrics_payload = {
            "job_name": args.job_name,
            "job_slot": args.job_slot,
            "actor_id": args.job_actor_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paths": collected_metrics,
        }
        metrics_path = args.metrics_json.resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        print(f"[METRICS] Wrote per-path runtime metrics to {metrics_path}", flush=True)

    exit_code = 0
    if error_count > 0:
        print(f"{error_count} error(s) logged to {error_log_path}", flush=True)
        exit_code = 2
    else:
        print("All scenes completed without logged errors.", flush=True)

    print(
        f"Processed {processed_scenes} scenes covering {overall_labels} labels.",
        flush=True,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
