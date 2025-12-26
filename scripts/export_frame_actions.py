#!/usr/bin/env python3
# Example:
#   python3 scripts/export_frame_actions.py /data1/33w_key2
# Example with optional tweaks:
#   python3 scripts/export_frame_actions.py /data1/33w_key2 --scenes-dir ./data/scenes --max-next 8 --move-threshold-deg 10 --turn-threshold-deg 15
from __future__ import annotations

import argparse
import json
import math
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
        "--output-name",
        type=str,
        default="frame_actions.json",
        help="Per-path output filename (default: frame_actions.json).",
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
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    max_next = max(0, int(args.max_next))

    outputs_written = 0

    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        meta = None
        scene_meta_dir = args.scenes_dir / scene_id
        try:
            meta = load_occupancy_metadata(scene_meta_dir)
        except FileNotFoundError:
            meta = None

        for label_dir in iter_label_dirs(scene_dir):
            camera_files = []
            for path in label_dir.iterdir():
                idx = parse_frame_index(path)
                if idx is not None:
                    camera_files.append((idx, path))
            if not camera_files:
                continue
            camera_files.sort(key=lambda item: item[0])

            frames = []
            for frame_idx, cam_path in camera_files:
                with cam_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                cam_center = payload.get("camera_center_world")
                cam_to_world = payload.get("camera_to_world")
                if not cam_center or not cam_to_world:
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
                continue

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
                ahead = dot > float(args.ahead_dot_eps)

                if angle_deg >= float(args.turn_threshold_deg):
                    actions.append("turn left" if delta_yaw > 0 else "turn right")
                elif angle_deg <= float(args.move_threshold_deg) and ahead:
                    actions.append("move")
                else:
                    actions.append("move" if ahead else "stop")

            for i, frame in enumerate(frames):
                next_actions = []
                for j in range(1, max_next + 1):
                    if i + j < len(actions):
                        next_actions.append(actions[i + j])
                    else:
                        next_actions.append("stop")

                per_frame.append(
                    {
                        "frame": frame["frame"],
                        "world": frame["world"],
                        "pixel": frame["pixel"],
                        "curr_action": actions[i],
                        "next_actions": next_actions,
                    }
                )

            payload = {
                "dataset_root": str(dataset_root),
                "scene": scene_id,
                "label": label_dir.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "max_next": max_next,
                "move_threshold_deg": float(args.move_threshold_deg),
                "turn_threshold_deg": float(args.turn_threshold_deg),
                "frames": per_frame,
            }
            output_path = label_dir / args.output_name
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            outputs_written += 1
    if outputs_written == 0:
        raise RuntimeError(f"No camera frame JSONs found under {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
