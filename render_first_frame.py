#!/usr/bin/env python3

"""Render first frame for raster_world trajectories."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import TextIO

import traceback
from datetime import datetime

import imageio.v2 as imageio
import numpy as np
import torch

from arguments import PipelineParams
from gaussian_renderer import render_or
from scene import GaussianModel

from utils.render_utils import build_perspective_camera, load_raster_world_points_only


JSON_DIR = Path("/home/tianhang/habitatbuild/habitat-sim/navdata_multi_terrain_pixelspace_with_gs")
PLY_PATH = Path("/home/tianhang/NavDP/navdp_api/gaussian_splatting/data/floor30geo_noncut_v1.ply")
DEFAULT_ERROR_LOG = JSON_DIR.parent / "errors.log"
EPS = 1e-6
CAMERA_HEIGHT_OFFSET = 1.3


def render_path_first_frame(
    json_path: Path,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    device: torch.device,
    bg_color: torch.Tensor,
    look_down: float,
    width: int,
    height: int,
    fov_deg: float,
    znear: float,
    zfar: float,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Render first frame for a raster_world trajectory."""

    raster_world = load_raster_world_points_only(json_path)
    if len(raster_world) < 2:
        raise ValueError(f"Need at least two points in {json_path}")

    first = raster_world[0]
    second = raster_world[1]

    position = np.array([first[0], first[1], first[2] + CAMERA_HEIGHT_OFFSET], dtype=np.float32)
    target = np.array([second[0], second[1], second[2] + CAMERA_HEIGHT_OFFSET], dtype=np.float32)

    top_down_map_dir = JSON_DIR / "gs_obs"
    top_down_map_dir.mkdir(parents=True, exist_ok=True)
    id_str = json_path.stem
    output_path = top_down_map_dir / f"start_rgb_gs_{id_str}.png"

    if not overwrite and output_path.exists():
        if verbose:
            print(f"Skipping {json_path.stem}: image already exists.")
        return

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

    with torch.no_grad():
        img_pkg = render_or(camera, gaussians, pipeline, bg_color=bg_color)
        render = img_pkg["render"].detach().cpu().numpy()
        render_uint8 = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
        render_uint8 = np.rot90(render_uint8, k=2)

        imageio.imwrite(output_path, render_uint8)

    if verbose:
        print(f"Rendered {json_path.stem} to {output_path}.")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Render first frame along raster_world navigation paths.")
    parser.add_argument(
        "--look-down",
        type=float,
        default=0.1,
        help="Vertical offset in meters to tilt the camera downward (default: 0.1).",
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
        "--overwrite",
        action="store_true",
        help="Re-render frame even if target image already exists.",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERROR_LOG,
        help=f"Append render failures to the specified log file (default: {DEFAULT_ERROR_LOG}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Output detailed rendering progress.",
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for rendering but not available.")
    device = torch.device("cuda")

    width = 256
    height = 256

    pipeline_parser = ArgumentParser(description="Pipeline parameters placeholder")
    pipeline = PipelineParams(pipeline_parser)
    pipeline.antialiasing = True
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    verbose_enabled = bool(args.verbose)

    # Load fixed PLY
    print(f"Loading point cloud: {PLY_PATH}", flush=True)
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(PLY_PATH))

    # Traverse all JSON files
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {JSON_DIR}.")
        return

    total_labels = len(json_files)
    print(f"Rendering first frame for {total_labels} paths.", flush=True)

    error_log_path: Path = args.error_log
    error_log_file: TextIO | None = None
    error_count = 0

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
        for label_idx, json_path in enumerate(json_files, start=1):
            print(f"[{label_idx}/{total_labels}] → {json_path.stem}", flush=True)
            try:
                render_path_first_frame(
                    json_path=json_path,
                    gaussians=gaussians,
                    pipeline=pipeline,
                    device=device,
                    bg_color=bg_color,
                    look_down=args.look_down,
                    width=width,
                    height=height,
                    fov_deg=args.fov_deg,
                    znear=args.znear,
                    zfar=args.zfar,
                    overwrite=args.overwrite,
                    verbose=verbose_enabled,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  ⚠️ Rendering {json_path.name} failed: {exc}")
                log_file = ensure_log_file()
                error_count += 1
                timestamp = datetime.now().isoformat(timespec="seconds")
                log_file.write(
                    f"[{timestamp}] Label={json_path.name} Error={exc}\n"
                )
                log_file.write(traceback.format_exc())
                log_file.write("\n")
                log_file.flush()
    finally:
        if error_log_file is not None:
            error_log_file.close()

    if error_count > 0:
        print(f"Processing complete, but {error_count} tasks failed; see {error_log_path} for details.", flush=True)
    else:
        print("All tasks complete, no errors detected.", flush=True)


if __name__ == "__main__":
    main()
