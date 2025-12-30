#!/usr/bin/env python3
"""Summarize navigation dataset statistics and emit charts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from analysis_utils import (
    DEFAULT_TASKS_DIR,
    FRAME_STEP_WORLD,
    FRAME_THRESHOLDS,
    describe,
    resolve_label_directory,
)


try:  # matplotlib is only needed for chart generation
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover - matplotlib is optional
    plt = None
    _HAVE_MPL = False


@dataclass(frozen=True)
class PathMetrics:
    scene_id: str
    label_id: str
    label_name: str | None
    file_path: Path
    path_length: float
    keypoint_count: int
    raster_steps: int | None
    displacement: float
    tortuosity: float | None
    estimated_frames: float


def load_path_metrics(scene_id: str, json_path: Path) -> PathMetrics | None:
    try:
        payload = json.loads(json_path.read_text())
    except Exception as exc:  # pragma: no cover - IO errors are recorded elsewhere
        raise RuntimeError(f"Failed to read {json_path}: {exc}") from exc

    path_section = payload.get("path", {}) or {}
    keypoints_world = path_section.get("keypoints_world") or []
    coords: list[tuple[float, float]] = []
    for entry in keypoints_world:
        x = entry.get("x")
        y = entry.get("y")
        if x is None or y is None:
            continue
        coords.append((float(x), float(y)))

    if len(coords) < 2:
        raise RuntimeError(f"Path {json_path} does not contain enough keypoints to measure length.")

    path_length = sum(
        math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords[:-1], coords[1:])
    )
    displacement = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
    tortuosity = (path_length / displacement) if displacement > 0 else None
    estimated_frames = path_length / FRAME_STEP_WORLD if FRAME_STEP_WORLD > 0 else float("nan")

    raster_steps = None
    raster_pixels = path_section.get("raster_pixel")
    if isinstance(raster_pixels, list):
        raster_steps = len(raster_pixels)

    label_payload = payload.get("label", {}) or {}
    label_id = str(label_payload.get("ins_id", json_path.stem))
    label_name = label_payload.get("label")

    return PathMetrics(
        scene_id=scene_id,
        label_id=label_id,
        label_name=label_name,
        file_path=json_path,
        path_length=path_length,
        keypoint_count=len(coords),
        raster_steps=raster_steps,
        displacement=displacement,
        tortuosity=tortuosity,
        estimated_frames=estimated_frames,
    )


def build_charts(
    *,
    output_dir: Path,
    path_lengths: list[float],
    scene_summaries: list[dict],
    top_scenes: int,
    hist_bins: int,
    frame_thresholds: dict[int, float],
) -> dict[str, str]:
    if not _HAVE_MPL:
        return {}

    charts: dict[str, str] = {}
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if path_lengths:
        plt.figure(figsize=(8, 4.5))
        plt.hist(path_lengths, bins=hist_bins, color="#3b82f6", edgecolor="black", alpha=0.8)
        plt.title("Path length distribution")
        plt.xlabel("Path length (world units)")
        plt.ylabel("Count")
        for frames, threshold_len in sorted(frame_thresholds.items()):
            plt.axvline(
                threshold_len,
                color="#ef4444",
                linestyle="--",
                linewidth=1.2,
                label=f">= {frames} frames",
            )
        if frame_thresholds:
            plt.legend(loc="upper right")
        plt.grid(alpha=0.2, linestyle="--")
        hist_path = charts_dir / "path_length_histogram.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        charts["path_length_histogram"] = str(hist_path)

    if scene_summaries:
        limited = sorted(scene_summaries, key=lambda item: item["path_count"], reverse=True)[:top_scenes]
        if limited:
            plt.figure(figsize=(10, 4.5))
            labels = [entry["scene_id"] for entry in limited]
            counts = [entry["path_count"] for entry in limited]
            plt.bar(range(len(limited)), counts, color="#10b981")
            plt.xticks(range(len(limited)), labels, rotation=60, ha="right", fontsize=8)
            plt.ylabel("Path count")
            plt.title(f"Top {len(limited)} scenes by path count")
            plt.tight_layout()
            top_path = charts_dir / "scene_path_count_top.png"
            plt.savefig(top_path, dpi=150)
            plt.close()
            charts["top_scenes_by_path_count"] = str(top_path)

        avg_lengths = [entry["path_length_stats"].get("mean") for entry in scene_summaries if entry["path_count"] > 0]
        counts_all = [entry["path_count"] for entry in scene_summaries if entry["path_count"] > 0]
        if avg_lengths and counts_all and len(avg_lengths) == len(counts_all):
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(counts_all, avg_lengths, alpha=0.5, color="#f97316")
            plt.xlabel("Path count per scene")
            plt.ylabel("Average path length (world units)")
            plt.title("Path count vs. average length per scene")
            plt.grid(alpha=0.2, linestyle=":")
            plt.tight_layout()
            scatter_path = charts_dir / "count_vs_avg_length.png"
            plt.savefig(scatter_path, dpi=150)
            plt.close()
            charts["path_count_vs_avg_length"] = str(scatter_path)

    return charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NavDP data generation tasks directory")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="Directory that contains per-scene task folders (default: ./data/selected_33w).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./analysis/datagen"),
        help="Directory where analysis JSON and charts will be written.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of bins for the path length histogram.",
    )
    parser.add_argument(
        "--top-scenes",
        type=int,
        default=20,
        help="Number of scenes to highlight in the top-N chart.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="datagen_analysis.json",
        help="Base filename for the JSON report.",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Optional limit on the number of scenes to scan (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_dir = args.tasks_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")

    per_scene_files: dict[str, list[Path]] = {}
    for scene_idx, scene_dir in enumerate(sorted(tasks_dir.iterdir()), start=1):
        if not scene_dir.is_dir():
            continue
        if args.limit_scenes and len(per_scene_files) >= args.limit_scenes:
            break
        label_dir = resolve_label_directory(scene_dir)
        if label_dir is None:
            continue
        json_files = sorted(label_dir.glob("*.json"))
        if json_files:
            per_scene_files[scene_dir.name] = json_files

    path_records: list[PathMetrics] = []
    failures: list[dict[str, str]] = []

    for scene_id, files in per_scene_files.items():
        for json_path in files:
            try:
                metrics = load_path_metrics(scene_id, json_path)
            except Exception as exc:  # pragma: no cover - logged for later review
                failures.append({"scene_id": scene_id, "file": str(json_path), "error": str(exc)})
                continue
            path_records.append(metrics)

    paths_by_scene: dict[str, list[PathMetrics]] = defaultdict(list)
    for record in path_records:
        paths_by_scene[record.scene_id].append(record)

    scene_summaries: list[dict] = []
    for scene_id in sorted(paths_by_scene):
        scene_paths = paths_by_scene[scene_id]
        lengths = [item.path_length for item in scene_paths]
        displacements = [item.displacement for item in scene_paths]
        raster_steps = [item.raster_steps for item in scene_paths if item.raster_steps is not None]
        tortuosity = [item.tortuosity for item in scene_paths if item.tortuosity is not None]
        thresholds_map: dict[str, int] = {}
        for frames in FRAME_THRESHOLDS:
            length_threshold = frames * FRAME_STEP_WORLD
            thresholds_map[str(frames)] = sum(1 for item in scene_paths if item.path_length >= length_threshold)
        summary = {
            "scene_id": scene_id,
            "path_count": len(scene_paths),
            "path_length_stats": describe(lengths),
            "displacement_stats": describe(displacements),
            "raster_step_stats": describe(raster_steps),
            "tortuosity_stats": describe(tortuosity),
            "paths_ge_frames": thresholds_map,
        }
        scene_summaries.append(summary)

    path_lengths = [item.path_length for item in path_records]
    displacements_all = [item.displacement for item in path_records]
    raster_steps_all = [item.raster_steps for item in path_records if item.raster_steps is not None]
    tortuosity_all = [item.tortuosity for item in path_records if item.tortuosity is not None]

    frame_threshold_details: list[dict] = []
    frame_threshold_len_map: dict[int, float] = {}
    for frames in FRAME_THRESHOLDS:
        threshold_len = frames * FRAME_STEP_WORLD
        frame_threshold_len_map[frames] = threshold_len
        total_paths_ge = sum(1 for item in path_records if item.path_length >= threshold_len)
        per_scene_counts = [summary["paths_ge_frames"].get(str(frames), 0) for summary in scene_summaries]
        frame_threshold_details.append(
            {
                "frames": frames,
                "min_length_world": threshold_len,
                "total_paths": total_paths_ge,
                "per_scene_stats": describe(per_scene_counts),
            }
        )

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks_dir": str(tasks_dir),
        "output_dir": str(output_dir),
        "scene_count": len(scene_summaries),
        "total_paths": len(path_records),
        "overall_path_length_stats": describe(path_lengths),
        "overall_displacement_stats": describe(displacements_all),
        "overall_raster_step_stats": describe(raster_steps_all),
        "overall_tortuosity_stats": describe(tortuosity_all),
        "scene_path_count_stats": describe([entry["path_count"] for entry in scene_summaries]),
        "per_scene": scene_summaries,
        "frame_thresholds": frame_threshold_details,
        "failures": failures,
    }

    longest_paths = sorted(path_records, key=lambda item: item.path_length, reverse=True)[:10]
    analysis["longest_paths"] = [
        {
            "scene_id": record.scene_id,
            "label_id": record.label_id,
            "label_name": record.label_name,
            "length": record.path_length,
            "file": str(record.file_path),
        }
        for record in longest_paths
    ]

    shortest_paths = sorted(path_records, key=lambda item: item.path_length)[:10]
    analysis["shortest_paths"] = [
        {
            "scene_id": record.scene_id,
            "label_id": record.label_id,
            "label_name": record.label_name,
            "length": record.path_length,
            "file": str(record.file_path),
        }
        for record in shortest_paths
    ]

    charts = build_charts(
        output_dir=output_dir,
        path_lengths=path_lengths,
        scene_summaries=scene_summaries,
        top_scenes=args.top_scenes,
        hist_bins=args.hist_bins,
        frame_thresholds=frame_threshold_len_map,
    )
    analysis["charts"] = charts

    json_path = output_dir / args.json_name
    json_path.write_text(json.dumps(analysis, indent=2))

    def fmt_stat(stat_key: str, stats_dict: dict) -> str:
        if stats_dict.get("count", 0) == 0:
            return "n/a"
        value = stats_dict.get(stat_key)
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    length_stats = analysis["overall_path_length_stats"]
    count_stats = analysis["scene_path_count_stats"]

    print(f"Analyzed {analysis['scene_count']} scene(s) with {analysis['total_paths']} path(s) from {tasks_dir}.")
    if analysis["scene_count"]:
        print(
            "Paths per scene -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", count_stats),
                median=fmt_stat("median", count_stats),
                mean=fmt_stat("mean", count_stats),
                max=fmt_stat("max", count_stats),
            )
        )
    if length_stats.get("count", 0):
        print(
            "Path length (world units) -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", length_stats),
                median=fmt_stat("median", length_stats),
                mean=fmt_stat("mean", length_stats),
                max=fmt_stat("max", length_stats),
            )
        )
    for threshold_entry in analysis["frame_thresholds"]:
        stats = threshold_entry["per_scene_stats"]
        if stats.get("count", 0) == 0:
            continue
        print(
            "Paths >= {frames} frames (~{length:.2f} units) -> total={total}, min={min}, mid={median}, ave={mean}, max={max}".format(
                frames=threshold_entry["frames"],
                length=threshold_entry["min_length_world"],
                total=threshold_entry["total_paths"],
                min=fmt_stat("min", stats),
                median=fmt_stat("median", stats),
                mean=fmt_stat("mean", stats),
                max=fmt_stat("max", stats),
            )
        )

    if charts:
        print("Charts written to: ")
        for label, path in charts.items():
            print(f"  - {label}: {path}")
    if failures:
        print(f"Encountered {len(failures)} unreadable/invalid path file(s); see JSON for details.")

    print(f"Full report: {json_path}")


if __name__ == "__main__":
    main()
