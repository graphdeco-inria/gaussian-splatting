#!/usr/bin/env python3
"""Analyze rendered datagen outputs and compare coverage against tasks."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from argparse import BooleanOptionalAction

from analysis_utils import (
    DEFAULT_RENDERS_DIR,
    DEFAULT_ASSIGNMENTS_JSON,
    FRAME_THRESHOLDS,
    FRAME_STEP_WORLD,
    describe,
    resolve_label_directory,
)

try:  # matplotlib is optional
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover - matplotlib not available
    plt = None
    _HAVE_MPL = False


@dataclass(frozen=True)
class RenderedPath:
    scene_id: str
    label_id: str
    frame_count: int
    path_length: float
    follow_distance: float
    metadata_path: Path
    video_path: Path
    has_video: bool
    video_size_bytes: int
    frames_dir: Path
    depth_frame_count: int
    rgb_frame_count: int


def _count_glob(directory: Path, pattern: str) -> int:
    return sum(1 for _ in directory.glob(pattern))


def load_rendered_path(scene_id: str, metadata_path: Path) -> RenderedPath:
    payload = json.loads(metadata_path.read_text())
    frames = payload.get("frames") or []
    frame_count = len(frames)
    person_coords: list[tuple[float, float]] = []
    for frame in frames:
        person_world = frame.get("person_world")
        if not isinstance(person_world, (list, tuple)):
            continue
        if len(person_world) < 2:
            continue
        try:
            person_coords.append((float(person_world[0]), float(person_world[1])))
        except (TypeError, ValueError):
            continue

    path_length = 0.0
    if len(person_coords) >= 2:
        path_length = sum(
            math.hypot(bx - ax, by - ay)
            for (ax, ay), (bx, by) in zip(person_coords[:-1], person_coords[1:])
        )

    follow_distance = float(payload.get("follow_distance", 0.0))
    label_id = str(payload.get("label", metadata_path.name.split("_follow_path", 1)[0]))
    video_path = metadata_path.with_name(f"{label_id}.mp4")
    has_video = video_path.is_file()
    video_size = video_path.stat().st_size if has_video else 0
    frames_dir = metadata_path.parent / label_id
    depth_count = _count_glob(frames_dir, "frame_*_depth.png") if frames_dir.is_dir() else 0
    rgb_count = _count_glob(frames_dir, "frame_*_rgb.png") if frames_dir.is_dir() else 0

    return RenderedPath(
        scene_id=scene_id,
        label_id=label_id,
        frame_count=frame_count,
        path_length=path_length,
        follow_distance=follow_distance,
        metadata_path=metadata_path,
        video_path=video_path,
        has_video=has_video,
        video_size_bytes=video_size,
        frames_dir=frames_dir,
        depth_frame_count=depth_count,
        rgb_frame_count=rgb_count,
    )


def collect_rendered_paths(renders_dir: Path, limit_scenes: int | None = None) -> tuple[list[RenderedPath], list[dict]]:
    rendered: list[RenderedPath] = []
    failures: list[dict] = []
    scene_seen = 0
    for scene_dir in sorted(renders_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_seen += 1
        if limit_scenes is not None and scene_seen > limit_scenes:
            break
        metadata_paths = sorted(scene_dir.glob("*_follow_path.json"))
        for metadata_path in metadata_paths:
            try:
                rendered.append(load_rendered_path(scene_dir.name, metadata_path))
            except Exception as exc:  # pragma: no cover - recorded for inspection
                failures.append({
                    "scene_id": scene_dir.name,
                    "file": str(metadata_path),
                    "error": str(exc),
                })
    return rendered, failures


def load_expected_labels(tasks_dir: Path) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    for scene_dir in sorted(tasks_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        label_dir = resolve_label_directory(scene_dir)
        if label_dir is None:
            continue
        json_files = [path for path in label_dir.glob("*.json")]
        if not json_files:
            continue
        expected[scene_dir.name] = {path.stem for path in json_files}
    return expected


def build_charts(
    *,
    report_dir: Path,
    path_lengths: list[float],
    frame_counts: list[int],
    scene_summaries: list[dict],
    top_scenes: int,
    hist_bins: int,
    frame_length_map: dict[int, float],
) -> dict[str, str]:
    if not _HAVE_MPL:
        return {}

    charts: dict[str, str] = {}
    charts_dir = report_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if path_lengths:
        plt.figure(figsize=(8, 4.5))
        plt.hist(path_lengths, bins=hist_bins, color="#2563eb", edgecolor="black", alpha=0.85)
        plt.title("Rendered path length distribution")
        plt.xlabel("Path length (world units)")
        plt.ylabel("Count")
        for frames, threshold_len in sorted(frame_length_map.items()):
            plt.axvline(
                threshold_len,
                color="#ef4444",
                linestyle="--",
                linewidth=1.2,
                label=f">= {frames} frames",
            )
        if frame_length_map:
            plt.legend(loc="upper right")
        plt.grid(alpha=0.2, linestyle="--")
        path_hist = charts_dir / "rendered_path_length_histogram.png"
        plt.tight_layout()
        plt.savefig(path_hist, dpi=150)
        plt.close()
        charts["path_length_histogram"] = str(path_hist)

    if frame_counts:
        plt.figure(figsize=(8, 4.5))
        plt.hist(frame_counts, bins=hist_bins, color="#059669", edgecolor="black", alpha=0.85)
        plt.title("Frame count distribution")
        plt.xlabel("Frame count")
        plt.ylabel("Count")
        for frames in FRAME_THRESHOLDS:
            plt.axvline(
                frames,
                color="#f97316",
                linestyle=":",
                linewidth=1.2,
                label=f">= {frames} frames",
            )
        plt.legend(loc="upper right")
        plt.grid(alpha=0.2, linestyle="--")
        frame_hist = charts_dir / "frame_count_histogram.png"
        plt.tight_layout()
        plt.savefig(frame_hist, dpi=150)
        plt.close()
        charts["frame_count_histogram"] = str(frame_hist)

    if scene_summaries:
        limited = sorted(scene_summaries, key=lambda entry: entry["path_count"], reverse=True)[:top_scenes]
        if limited:
            plt.figure(figsize=(10, 4.5))
            labels = [entry["scene_id"] for entry in limited]
            counts = [entry["path_count"] for entry in limited]
            plt.bar(range(len(limited)), counts, color="#9333ea")
            plt.xticks(range(len(limited)), labels, rotation=60, ha="right", fontsize=8)
            plt.ylabel("Rendered paths")
            plt.title(f"Top {len(limited)} scenes by rendered paths")
            plt.tight_layout()
            top_chart = charts_dir / "rendered_scene_path_count_top.png"
            plt.savefig(top_chart, dpi=150)
            plt.close()
            charts["top_scenes_by_paths"] = str(top_chart)

        avg_frames = [entry["frame_count_stats"].get("mean") for entry in scene_summaries if entry["path_count"] > 0]
        counts_all = [entry["path_count"] for entry in scene_summaries if entry["path_count"] > 0]
        if avg_frames and counts_all and len(avg_frames) == len(counts_all):
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(counts_all, avg_frames, color="#0ea5e9", alpha=0.6)
            plt.xlabel("Paths per scene")
            plt.ylabel("Average frames")
            plt.title("Scene density vs. average frame count")
            plt.grid(alpha=0.2, linestyle=":")
            plt.tight_layout()
            scatter_chart = charts_dir / "scene_path_count_vs_avg_frames.png"
            plt.savefig(scatter_chart, dpi=150)
            plt.close()
            charts["scene_path_count_vs_avg_frames"] = str(scatter_chart)

    return charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rendered datagen outputs for quality/coverage metrics.")
    parser.add_argument(
        "--renders-dir",
        type=Path,
        default=DEFAULT_RENDERS_DIR,
        help="Root directory containing rendered scene folders (default: ./data/path_video_frames_random_humans_33w).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("./analysis/post_datagen"),
        help="Directory where JSON report and charts will be stored.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Optional path to the source tasks directory for coverage comparison.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "--top-scenes",
        type=int,
        default=20,
        help="Number of scenes to highlight in top-N charts.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="post_datagen_analysis.json",
        help="Filename for the output JSON report.",
    )
    parser.add_argument(
        "--assignments-json",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_JSON,
        help="Optional actor assignment manifest to map scenes/labels to avatars (default: ./data/actor_assignments_w_ban_33w.json).",
    )
    parser.add_argument(
        "--video-min-mb",
        type=float,
        default=1.0,
        help="Minimum acceptable MP4 size in megabytes before flagging as suspicious (default: 1 MB).",
    )
    parser.add_argument(
        "--check-depth",
        action=BooleanOptionalAction,
        default=True,
        help="Verify that every rendered path has depth PNG frames for each timestep (default: on).",
    )
    parser.add_argument(
        "--check-rgb",
        action=BooleanOptionalAction,
        default=True,
        help="Verify that every rendered path has RGB PNG frames for each timestep (default: on).",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Optional limit on the number of scene folders to scan (for quick smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renders_dir = args.renders_dir.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not renders_dir.is_dir():
        raise SystemExit(f"Renders directory not found: {renders_dir}")

    rendered_paths, failures = collect_rendered_paths(renders_dir, args.limit_scenes)
    paths_by_scene: dict[str, list[RenderedPath]] = defaultdict(list)
    for record in rendered_paths:
        paths_by_scene[record.scene_id].append(record)

    assignments_path: Path | None = None
    assignment_lookup: dict[tuple[str, str], dict] = {}
    if args.assignments_json:
        assignments_path = args.assignments_json.resolve()
        if not assignments_path.is_file():
            raise SystemExit(f"Assignments file not found: {assignments_path}")
        assignments_payload = json.loads(assignments_path.read_text())
        for entry in assignments_payload.get("assignments", []):
            scene = str(entry.get("scene"))
            label = str(entry.get("label"))
            assignment_lookup[(scene, label)] = entry

    scene_summaries: list[dict] = []
    missing_videos: list[dict[str, str]] = []
    for scene_id in sorted(paths_by_scene):
        entries = paths_by_scene[scene_id]
        frame_counts = [entry.frame_count for entry in entries]
        lengths = [entry.path_length for entry in entries]
        threshold_map = {
            str(frames): sum(1 for entry in entries if entry.frame_count >= frames)
            for frames in FRAME_THRESHOLDS
        }
        missing_labels = [entry.label_id for entry in entries if not entry.has_video]
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "path_count": len(entries),
                "frame_count_stats": describe(frame_counts),
                "path_length_stats": describe(lengths),
                "paths_ge_frames": threshold_map,
                "missing_video_labels": missing_labels,
                "missing_video_count": len(missing_labels),
                "problem_path_count": 0,
            }
        )
        for label in missing_labels:
            missing_videos.append({"scene_id": scene_id, "label_id": label})

    scene_summary_lookup = {entry["scene_id"]: entry for entry in scene_summaries}

    path_lengths = [entry.path_length for entry in rendered_paths]
    frame_counts_all = [entry.frame_count for entry in rendered_paths]
    per_scene_counts = [summary["path_count"] for summary in scene_summaries]

    frame_threshold_details: list[dict] = []
    frame_length_map: dict[int, float] = {}
    for frames in FRAME_THRESHOLDS:
        min_length = frames * FRAME_STEP_WORLD
        frame_length_map[frames] = min_length
        total_paths_ge = sum(1 for entry in rendered_paths if entry.frame_count >= frames)
        per_scene_ge = [summary["paths_ge_frames"].get(str(frames), 0) for summary in scene_summaries]
        frame_threshold_details.append(
            {
                "frames": frames,
                "min_length_world": min_length,
                "total_paths": total_paths_ge,
                "per_scene_stats": describe(per_scene_ge),
            }
        )

    video_min_bytes = max(int(args.video_min_mb * 1024 * 1024), 0)

    problem_paths: list[dict] = []
    for record in rendered_paths:
        issues: list[str] = []
        if not record.has_video:
            issues.append("missing_video")
        elif record.video_size_bytes < video_min_bytes:
            issues.append("small_video")
        if not record.frames_dir.exists():
            issues.append("frames_dir_missing")
        else:
            if args.check_depth and record.frame_count > record.depth_frame_count:
                issues.append("missing_depth_frames")
            if args.check_rgb and record.frame_count > record.rgb_frame_count:
                issues.append("missing_rgb_frames")
        if issues:
            actor_info = assignment_lookup.get((record.scene_id, record.label_id)) if assignment_lookup else None
            problem_entry = {
                "scene_id": record.scene_id,
                "label_id": record.label_id,
                "issues": issues,
                "frame_count": record.frame_count,
                "depth_frames": record.depth_frame_count,
                "rgb_frames": record.rgb_frame_count,
                "video_size_bytes": record.video_size_bytes,
                "video_path": str(record.video_path),
                "frames_dir": str(record.frames_dir),
                "assignment": actor_info,
            }
            problem_paths.append(problem_entry)
            if record.scene_id in scene_summary_lookup:
                scene_summary_lookup[record.scene_id]["problem_path_count"] += 1

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "renders_dir": str(renders_dir),
        "report_dir": str(report_dir),
        "scene_count": len(scene_summaries),
        "total_paths": len(rendered_paths),
        "overall_frame_count_stats": describe(frame_counts_all),
        "overall_path_length_stats": describe(path_lengths),
        "scene_path_count_stats": describe(per_scene_counts),
        "per_scene": scene_summaries,
        "frame_thresholds": frame_threshold_details,
        "video_missing_total": len(missing_videos),
        "video_missing": missing_videos,
        "failures": failures,
        "problem_path_count": len(problem_paths),
        "problem_paths": problem_paths,
    }
    if assignments_path:
        analysis["assignments_path"] = str(assignments_path)

    if args.tasks_dir:
        tasks_dir = args.tasks_dir.resolve()
        if not tasks_dir.is_dir():
            raise SystemExit(f"Tasks directory not found: {tasks_dir}")
        expected_labels = load_expected_labels(tasks_dir)
        rendered_scene_ids = set(paths_by_scene)
        expected_scene_ids = set(expected_labels)
        missing_scenes = sorted(expected_scene_ids - rendered_scene_ids)
        extra_scenes = sorted(rendered_scene_ids - expected_scene_ids)

        coverage_entries: list[dict] = []
        coverage_values: list[float] = []
        for scene_id in sorted(expected_scene_ids):
            expected = expected_labels[scene_id]
            produced = {entry.label_id for entry in paths_by_scene.get(scene_id, [])}
            overlap = len(expected & produced)
            coverage_ratio = overlap / len(expected) if expected else 1.0
            if expected:
                coverage_values.append(coverage_ratio)
            missing_labels = sorted(expected - produced)
            extra_labels = sorted(produced - expected)
            if missing_labels or extra_labels:
                coverage_entries.append(
                    {
                        "scene_id": scene_id,
                        "expected": len(expected),
                        "rendered": len(produced),
                        "coverage_ratio": coverage_ratio,
                        "missing_label_count": len(missing_labels),
                        "missing_label_examples": missing_labels[:20],
                        "extra_label_count": len(extra_labels),
                        "extra_label_examples": extra_labels[:20],
                    }
                )

        comparison = {
            "tasks_dir": str(tasks_dir),
            "expected_scene_count": len(expected_scene_ids),
            "missing_scene_count": len(missing_scenes),
            "extra_scene_count": len(extra_scenes),
            "missing_scenes": missing_scenes,
            "extra_scenes": extra_scenes,
            "coverage_ratio_stats": describe(coverage_values),
            "scenes_with_issues": coverage_entries,
        }
        analysis["comparison"] = comparison

    charts = build_charts(
        report_dir=report_dir,
        path_lengths=path_lengths,
        frame_counts=frame_counts_all,
        scene_summaries=scene_summaries,
        top_scenes=args.top_scenes,
        hist_bins=args.hist_bins,
        frame_length_map=frame_length_map,
    )
    analysis["charts"] = charts

    json_path = report_dir / args.json_name
    json_path.write_text(json.dumps(analysis, indent=2))

    def fmt_stat(stat_key: str, stats_dict: dict) -> str:
        if stats_dict.get("count", 0) == 0:
            return "n/a"
        value = stats_dict.get(stat_key)
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    count_stats = analysis["scene_path_count_stats"]
    length_stats = analysis["overall_path_length_stats"]
    frame_stats = analysis["overall_frame_count_stats"]

    print(
        f"Analyzed {analysis['scene_count']} scene(s) with {analysis['total_paths']} rendered path(s) from {renders_dir}."
    )
    if analysis["scene_count"]:
        print(
            "Rendered paths per scene -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", count_stats),
                median=fmt_stat("median", count_stats),
                mean=fmt_stat("mean", count_stats),
                max=fmt_stat("max", count_stats),
            )
        )
    if length_stats.get("count", 0):
        print(
            "Rendered path length (world units) -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", length_stats),
                median=fmt_stat("median", length_stats),
                mean=fmt_stat("mean", length_stats),
                max=fmt_stat("max", length_stats),
            )
        )
    if frame_stats.get("count", 0):
        print(
            "Frame count -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", frame_stats),
                median=fmt_stat("median", frame_stats),
                mean=fmt_stat("mean", frame_stats),
                max=fmt_stat("max", frame_stats),
            )
        )

    for threshold_entry in analysis["frame_thresholds"]:
        stats = threshold_entry["per_scene_stats"]
        if stats.get("count", 0) == 0:
            continue
        print(
            "Rendered paths >= {frames} frames (~{length:.2f} units) -> total={total}, min={min}, mid={median}, ave={mean}, max={max}".format(
                frames=threshold_entry["frames"],
                length=threshold_entry["min_length_world"],
                total=threshold_entry["total_paths"],
                min=fmt_stat("min", stats),
                median=fmt_stat("median", stats),
                mean=fmt_stat("mean", stats),
                max=fmt_stat("max", stats),
            )
        )

    if analysis.get("comparison"):
        comparison = analysis["comparison"]
        print(
            f"Coverage vs {comparison['tasks_dir']} -> missing scenes: {comparison['missing_scene_count']}, extra scenes: {comparison['extra_scene_count']}."
        )
        coverage_stats = comparison.get("coverage_ratio_stats", {})
        if coverage_stats.get("count", 0):
            print(
                "Coverage ratio -> min={min}, mid={median}, ave={mean}, max={max}".format(
                    min=fmt_stat("min", coverage_stats),
                    median=fmt_stat("median", coverage_stats),
                    mean=fmt_stat("mean", coverage_stats),
                    max=fmt_stat("max", coverage_stats),
                )
            )

    if analysis["video_missing_total"]:
        print(f"Video missing for {analysis['video_missing_total']} rendered path(s).")
    if analysis["problem_path_count"]:
        print(f"Detected {analysis['problem_path_count']} path(s) with output issues (see JSON for actor details).")
        for sample in analysis["problem_paths"][: min(5, analysis["problem_path_count"])]:
            actor_id = sample.get("assignment", {}).get("actor_id") if sample.get("assignment") else "unknown"
            print(
                f"  - {sample['scene_id']}/{sample['label_id']} actor={actor_id} issues={','.join(sample['issues'])}"
            )
    if failures:
        print(f"Encountered {len(failures)} unreadable metadata file(s); see JSON for details.")
    if charts:
        print("Charts written to:")
        for label, path in charts.items():
            print(f"  - {label}: {path}")

    print(f"Full report: {json_path}")


if __name__ == "__main__":
    main()
