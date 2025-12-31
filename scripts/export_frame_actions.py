#!/usr/bin/env python3
# Example:
#   python3 scripts/export_frame_actions.py ./data1/33w_key2
# Example with optional tweaks:
#   python3 scripts/export_frame_actions.py ./data1/33w_key2 --scenes-dir ./data/scenes --max-next 8 --move-threshold-deg 10 --turn-threshold-deg 15 --clean
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


FRAME_RE = re.compile(r"^frame_(\d+)_camera\.json$")


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


def load_occupancy_metadata(scene_dir: Path) -> dict:
    occ_json = scene_dir / "occupancy.json"
    if not occ_json.is_file():
        raise FileNotFoundError(f"Missing occupancy.json in {scene_dir}")
    with occ_json.open("r", encoding="utf-8") as fh:
        occ = json.load(fh)

    scale = float(occ.get("scale", 1.0))
    min_x, min_y, min_z = map(float, occ.get("min", (0.0, 0.0, 0.0)))
    max_x, max_y, max_z = map(float, occ.get("max", (0.0, 0.0, 0.0)))

    lower = occ.get("lower") or [min_x, min_y, min_z]
    upper = occ.get("upper") or [max_x, max_y, max_z]
    lower_z = float(lower[2])
    upper_z = float(upper[2])

    occ_png = scene_dir / "occupancy.png"
    if not occ_png.is_file():
        raise FileNotFoundError(f"Missing occupancy.png in {scene_dir}")
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


def world_to_pixel(meta: dict, x: float, y: float) -> tuple[int, int]:
    u = int(round((x - float(meta["left"])) / float(meta["scale"])))
    v = int(round((float(meta["top"]) - y) / float(meta["scale"])))
    return u, v


def parse_frame_index(path: Path) -> int | None:
    match = FRAME_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def iter_label_dirs(scene_dir: Path) -> Iterable[Path]:
    for child in sorted(scene_dir.iterdir()):
        if child.is_dir():
            yield child


def signed_angle_delta(a: float, b: float) -> float:
    delta = b - a
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def compute_yaw_delta_series(frames: list[dict], step: int) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for k in range(len(frames) - step):
        idx = int(frames[k]["frame"])
        y0 = float(frames[k]["yaw"])
        y1 = float(frames[k + step]["yaw"])
        deg = abs(math.degrees(signed_angle_delta(y0, y1)))
        xs.append(idx)
        ys.append(deg)
    return xs, ys


def render_plot(
    xs: list[int],
    ys: list[float],
    title: str,
    out_path: Path,
    mpl_config_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel("Yaw delta (degrees)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_scene(
    scene_dir: Path,
    scenes_meta_dir: Path,
    output_template: str,
    plot_name: str,
    max_next: int,
    skip_frames: int,
    move_threshold_deg: float,
    turn_threshold_deg: float,
    ahead_dot_eps: float,
    verbose: bool,
    plots: bool,
    clean: bool,
) -> tuple[str, int, int]:
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    scene_id = scene_dir.name
    meta = None
    scene_meta_dir = scenes_meta_dir / scene_id
    try:
        meta = load_occupancy_metadata(scene_meta_dir)
        log(f"[scene] {scene_id}: loaded occupancy metadata from {scene_meta_dir}")
    except FileNotFoundError:
        meta = None
        log(f"[scene] {scene_id}: occupancy metadata missing at {scene_meta_dir}")

    outputs_written = 0
    labels_seen = 0
    step = max(1, int(skip_frames) + 1)

    plots_root = scene_dir.parent / "plots"
    if clean:
        legacy_plot_scene = plots_root / scene_id
        if legacy_plot_scene.exists():
            for child in legacy_plot_scene.glob("*"):
                if child.is_file():
                    child.unlink()

    for label_dir in iter_label_dirs(scene_dir):
        labels_seen += 1
        output_name = output_template.replace("{label}", label_dir.name)
        output_path = scene_dir / output_name
        if clean:
            legacy_json = label_dir / "frame_actions.json"
            if legacy_json.exists():
                legacy_json.unlink()
            if output_path.exists():
                output_path.unlink()

        camera_files = []
        for path in label_dir.iterdir():
            idx = parse_frame_index(path)
            if idx is not None:
                camera_files.append((idx, path))
        if not camera_files:
            log(f"[label] {scene_id}/{label_dir.name}: no camera frames found")
            continue
        camera_files.sort(key=lambda item: item[0])

        frames = []
        for frame_idx, cam_path in camera_files:
            with cam_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            cam_center = payload.get("camera_center_world")
            cam_to_world = payload.get("camera_to_world")
            if not cam_center or not cam_to_world:
                log(f"[frame] {scene_id}/{label_dir.name}/{cam_path.name}: missing camera data")
                continue

            x, y, z = map(float, cam_center[:3])
            forward_row = cam_to_world[2]
            fx = float(forward_row[0])
            fy = float(forward_row[1])
            norm = math.hypot(fx, fy)
            if norm < 1e-6:
                fx, fy = 0.0, 1.0
            else:
                fx, fy = fx / norm, fy / norm
            yaw = math.atan2(fy, fx)

            pixel = None
            if meta is not None:
                u, v = world_to_pixel(meta, x, y)
                pixel = [int(u), int(v)]

            frames.append(
                {
                    "frame": int(frame_idx),
                    "world": [x, y, z],
                    "pixel": pixel,
                    "forward": [fx, fy],
                    "yaw": yaw,
                }
            )

        if len(frames) < 1:
            log(f"[label] {scene_id}/{label_dir.name}: no usable frames after parsing")
            continue
        log(f"[label] {scene_id}/{label_dir.name}: parsed {len(frames)} frames")

        actions: list[str] = []
        per_frame: list[dict] = []
        for i, frame in enumerate(frames):
            if i == len(frames) - 1:
                actions.append("stop")
                continue

            curr = frame
            nxt = frames[i + 1]
            delta_yaw = signed_angle_delta(curr["yaw"], nxt["yaw"])
            angle_deg = abs(math.degrees(delta_yaw))

            dx = nxt["world"][0] - curr["world"][0]
            dy = nxt["world"][1] - curr["world"][1]
            dot = dx * curr["forward"][0] + dy * curr["forward"][1]
            ahead = dot > float(ahead_dot_eps)

            if angle_deg >= float(turn_threshold_deg):
                actions.append("turn left" if delta_yaw > 0 else "turn right")
            elif angle_deg <= float(move_threshold_deg) and ahead:
                actions.append("move")
            else:
                actions.append("move" if ahead else "stop")

        action_codes = {
            "stop": 0,
            "move": 1,
            "turn left": 2,
            "turn right": 3,
        }

        for i, frame in enumerate(frames):
            next_actions = []
            for j in range(1, max_next + 1):
                if i + j < len(actions):
                    next_actions.append(action_codes[actions[i + j]])
                else:
                    next_actions.append(action_codes["stop"])

            per_frame.append(
                {
                    "frame": frame["frame"],
                    "world": frame["world"],
                    "pixel": frame["pixel"],
                    "curr_action": action_codes[actions[i]],
                    "next_actions": next_actions,
                }
            )

        payload = {
            "dataset_root": str(scene_dir.parent),
            "scene": scene_id,
            "label": label_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "max_next": max_next,
            "move_threshold_deg": float(move_threshold_deg),
            "turn_threshold_deg": float(turn_threshold_deg),
            "frames": per_frame,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        outputs_written += 1
        log(f"[write] {output_path}")

        if plots and len(frames) > step:
            plot_name_expanded = plot_name
            if "{skip}" in plot_name_expanded:
                plot_name_expanded = plot_name_expanded.replace("{skip}", str(skip_frames))
            plot_name_final = (
                plot_name_expanded.replace("{label}", label_dir.name)
                if "{label}" in plot_name_expanded
                else plot_name_expanded
            )
            plot_path = plots_root / scene_id / plot_name_final
            xs, ys = compute_yaw_delta_series(frames, step)
            title = f"Yaw Delta vs Frame (skip {skip_frames} frames)"
            mpl_config_dir = scene_dir.parent / ".mplconfig"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            render_plot(xs, ys, title, plot_path, mpl_config_dir)
            log(f"[plot] {plot_path}")

    return scene_id, labels_seen, outputs_written


def process_scene_args(args: tuple) -> tuple[str, int, int]:
    return process_scene(*args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export per-frame actions from a rendered dataset."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory of the rendered dataset (scene folders underneath).",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="{label}_actions.json",
        help="Output filename template placed under each scene (default: {label}_actions.json).",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="yaw_delta_{label}_skip{skip}.png",
        help="Per-path plot filename; supports {label} and {skip} placeholders.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "scenes",
        help="Directory containing scene occupancy metadata.",
    )
    parser.add_argument(
        "--max-next",
        type=int,
        default=8,
        help="Number of future actions to include per frame (default: 8).",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=4,
        help="Number of frames to skip when plotting yaw deltas (default: 4).",
    )
    parser.add_argument(
        "--move-threshold-deg",
        type=float,
        default=10.0,
        help="Max yaw change for a move action (default: 10).",
    )
    parser.add_argument(
        "--turn-threshold-deg",
        type=float,
        default=15.0,
        help="Min yaw change for a turn action (default: 15).",
    )
    parser.add_argument(
        "--ahead-dot-eps",
        type=float,
        default=1e-6,
        help="Minimum forward dot-product to consider the next frame ahead.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel scene workers (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable plot generation (default: off).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous outputs before writing new results.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    max_next = max(0, int(args.max_next))

    skip_scene_names = {"plots", ".mplconfig"}
    scene_dirs = [
        p
        for p in sorted(dataset_root.iterdir())
        if p.is_dir() and p.name not in skip_scene_names
    ]
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {dataset_root}")

    if args.verbose:
        print(
            f"[start] scenes={len(scene_dirs)} workers={args.workers} skip_frames={args.skip_frames} plots={args.plots}",
            flush=True,
        )

    from multiprocessing import Pool

    total_labels = 0
    outputs_written = 0
    work_items = [
        (
            scene_dir,
            args.scenes_dir,
            args.output_template,
            args.plot_name,
            max_next,
            args.skip_frames,
            args.move_threshold_deg,
            args.turn_threshold_deg,
            args.ahead_dot_eps,
            args.verbose,
            args.plots,
            args.clean,
        )
        for scene_dir in scene_dirs
    ]

    with Pool(processes=args.workers) as pool:
        for scene_id, labels_seen, written in pool.imap_unordered(process_scene_args, work_items):
            total_labels += labels_seen
            outputs_written += written
            if args.verbose:
                print(
                    f"[progress] scene={scene_id} labels={labels_seen} outputs={written}",
                    flush=True,
                )

    if outputs_written == 0:
        raise RuntimeError(f"No camera frame JSONs found under {dataset_root}")
    if args.verbose:
        print(
            f"[summary] scenes={len(scene_dirs)} labels={total_labels} outputs={outputs_written}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
