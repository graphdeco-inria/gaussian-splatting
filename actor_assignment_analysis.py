#!/usr/bin/env python3
"""Summarize actor assignment manifests and compare against tasks if provided."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analysis_utils import (
    DEFAULT_ASSIGNMENTS_JSON,
    DEFAULT_TASKS_DIR,
    describe,
    resolve_label_directory,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    plt = None
    _HAVE_MPL = False


def load_assignments(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text())
    return payload.get("actors", []), payload.get("assignments", [])


def load_tasks(tasks_dir: Path) -> dict[str, set[str]]:
    scene_map: dict[str, set[str]] = {}
    for scene_dir in sorted(tasks_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        label_dir = resolve_label_directory(scene_dir)
        if label_dir is None:
            continue
        json_files = [path.stem for path in label_dir.glob("*.json")]
        if json_files:
            scene_map[scene_dir.name] = set(json_files)
    return scene_map


def build_charts(
    *,
    output_dir: Path,
    scene_counts: list[int],
    actor_usage: list[int],
    top_actors: list[tuple[str, int]],
    hist_bins: int,
) -> dict[str, str]:
    if not _HAVE_MPL:
        return {}
    charts: dict[str, str] = {}
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if scene_counts:
        plt.figure(figsize=(8, 4.5))
        plt.hist(scene_counts, bins=hist_bins, color="#2563eb", edgecolor="black", alpha=0.85)
        plt.title("Assignments per scene")
        plt.xlabel("Assignments")
        plt.ylabel("Scene count")
        plt.grid(alpha=0.2, linestyle="--")
        scene_hist = charts_dir / "scene_assignment_histogram.png"
        plt.tight_layout()
        plt.savefig(scene_hist, dpi=150)
        plt.close()
        charts["scene_assignment_histogram"] = str(scene_hist)

    if actor_usage:
        plt.figure(figsize=(8, 4.5))
        plt.hist(actor_usage, bins=hist_bins, color="#059669", edgecolor="black", alpha=0.85)
        plt.title("Assignments per actor")
        plt.xlabel("Assignments")
        plt.ylabel("Actor count")
        plt.grid(alpha=0.2, linestyle="--")
        actor_hist = charts_dir / "actor_usage_histogram.png"
        plt.tight_layout()
        plt.savefig(actor_hist, dpi=150)
        plt.close()
        charts["actor_usage_histogram"] = str(actor_hist)

    if top_actors:
        plt.figure(figsize=(10, 4.5))
        labels = [entry[0] for entry in top_actors]
        counts = [entry[1] for entry in top_actors]
        plt.bar(range(len(top_actors)), counts, color="#f97316")
        plt.xticks(range(len(top_actors)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Assignments")
        plt.title(f"Top {len(top_actors)} actors by usage")
        plt.tight_layout()
        top_chart = charts_dir / "top_actor_usage.png"
        plt.savefig(top_chart, dpi=150)
        plt.close()
        charts["top_actor_usage"] = str(top_chart)

    return charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze actor assignment manifest for datagen.")
    parser.add_argument(
        "--assignments-json",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_JSON,
        help="Path to actor assignment JSON (default: ./data/actor_assignments_w_ban_33w.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./analysis/actor_assignments"),
        help="Directory where analysis JSON/charts will be written.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "--top-actors",
        type=int,
        default=20,
        help="Top N actors to display in usage chart.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Optional tasks directory for scene/label coverage comparison.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="actor_assignments_analysis.json",
        help="Filename for the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments_path = args.assignments_json.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not assignments_path.is_file():
        raise SystemExit(f"Assignments file not found: {assignments_path}")

    actors, assignments = load_assignments(assignments_path)
    actor_meta = {entry.get("id"): entry for entry in actors}

    print(f"Analyzing assignments from {assignments_path} ...")

    scene_counter: Counter[str] = Counter()
    actor_counter: Counter[str] = Counter()
    round_counter: Counter[int] = Counter()
    height_samples: list[float] = []
    foot_offsets: list[float] = []
    speed_samples: list[float] = []
    fps_samples: list[float] = []

    for entry in assignments:
        scene = entry.get("scene")
        actor_id = entry.get("actor_id")
        scene_counter[scene] += 1
        actor_counter[actor_id] += 1
        round_counter[int(entry.get("round", 0))] += 1
        if entry.get("actor_height") is not None:
            height_samples.append(float(entry["actor_height"]))
        if entry.get("actor_foot_offset") is not None:
            foot_offsets.append(float(entry["actor_foot_offset"]))
        actor_info = actor_meta.get(actor_id)
        if actor_info:
            if actor_info.get("speed") is not None:
                speed_samples.append(float(actor_info["speed"]))
            if actor_info.get("fps") is not None:
                fps_samples.append(float(actor_info["fps"]))

    scene_stats = describe(scene_counter.values())
    actor_usage_stats = describe(actor_counter.values())
    round_stats = {str(round_idx): count for round_idx, count in sorted(round_counter.items())}

    tasks_comparison = None
    if args.tasks_dir:
        tasks_dir = args.tasks_dir.resolve()
        if not tasks_dir.is_dir():
            raise SystemExit(f"Tasks directory not found: {tasks_dir}")
        expected = load_tasks(tasks_dir)
        assigned_scenes = set(scene_counter)
        expected_scenes = set(expected)
        missing_scenes = sorted(expected_scenes - assigned_scenes)
        extra_scenes = sorted(assigned_scenes - expected_scenes)
        coverage_entries: list[dict] = []
        coverage_values: list[float] = []
        for scene_id, scene_labels in expected.items():
            assigned_labels = {entry["label"] for entry in assignments if entry.get("scene") == scene_id}
            overlap = len(scene_labels & assigned_labels)
            ratio = overlap / len(scene_labels) if scene_labels else 1.0
            if scene_labels:
                coverage_values.append(ratio)
            if overlap != len(scene_labels):
                missing_labels = sorted(scene_labels - assigned_labels)
                coverage_entries.append(
                    {
                        "scene_id": scene_id,
                        "expected_labels": len(scene_labels),
                        "assigned_labels": len(assigned_labels),
                        "coverage_ratio": ratio,
                        "missing_label_count": len(missing_labels),
                        "missing_samples": missing_labels[:25],
                    }
                )
        tasks_comparison = {
            "tasks_dir": str(tasks_dir),
            "expected_scene_count": len(expected_scenes),
            "assigned_scene_count": len(assigned_scenes),
            "missing_scene_count": len(missing_scenes),
            "extra_scene_count": len(extra_scenes),
            "missing_scenes": missing_scenes,
            "extra_scenes": extra_scenes,
            "coverage_ratio_stats": describe(coverage_values),
            "coverage_issues": coverage_entries,
        }

    scene_counts = list(scene_counter.values())
    actor_usage = list(actor_counter.values())
    top_usage = sorted(actor_counter.items(), key=lambda item: item[1], reverse=True)[: args.top_actors]

    charts = build_charts(
        output_dir=output_dir,
        scene_counts=scene_counts,
        actor_usage=actor_usage,
        top_actors=top_usage,
        hist_bins=args.hist_bins,
    )

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assignments_path": str(assignments_path),
        "scene_count": len(scene_counter),
        "assignment_count": len(assignments),
        "actor_count": len(actor_meta),
        "actors_used": len(actor_counter),
        "scene_assignment_stats": scene_stats,
        "actor_usage_stats": actor_usage_stats,
        "round_counts": round_stats,
        "height_stats": describe(height_samples),
        "foot_offset_stats": describe(foot_offsets),
        "speed_stats": describe(speed_samples),
        "fps_stats": describe(fps_samples),
        "charts": charts,
    }
    if tasks_comparison:
        analysis["tasks_comparison"] = tasks_comparison
        analysis["tasks_dir"] = tasks_comparison["tasks_dir"]

    json_path = output_dir / args.json_name
    json_path.write_text(json.dumps(analysis, indent=2))

    def fmt_stat(label: str, stats: dict) -> str:
        if stats.get("count", 0) == 0:
            return "n/a"
        value = stats.get(label)
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    print(
        f"Loaded {analysis['assignment_count']} assignment(s) across {analysis['scene_count']} scene(s); "
        f"{analysis['actors_used']} / {analysis['actor_count']} actors used."
    )
    if scene_stats.get("count", 0):
        print(
            "Assignments per scene -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", scene_stats),
                median=fmt_stat("median", scene_stats),
                mean=fmt_stat("mean", scene_stats),
                max=fmt_stat("max", scene_stats),
            )
        )
    if actor_usage_stats.get("count", 0):
        print(
            "Assignments per actor -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", actor_usage_stats),
                median=fmt_stat("median", actor_usage_stats),
                mean=fmt_stat("mean", actor_usage_stats),
                max=fmt_stat("max", actor_usage_stats),
            )
        )
    if tasks_comparison:
        coverage_stats = tasks_comparison.get("coverage_ratio_stats", {})
        print(
            f"Scene coverage vs {tasks_comparison['tasks_dir']} -> missing {tasks_comparison['missing_scene_count']} scene(s), extra {tasks_comparison['extra_scene_count']} scene(s)."
        )
        if coverage_stats.get("count", 0):
            print(
                "Label coverage ratio -> min={min}, mid={median}, ave={mean}, max={max}".format(
                    min=fmt_stat("min", coverage_stats),
                    median=fmt_stat("median", coverage_stats),
                    mean=fmt_stat("mean", coverage_stats),
                    max=fmt_stat("max", coverage_stats),
                )
            )

    if charts:
        print("Charts written to:")
        for label, path in charts.items():
            print(f"  - {label}: {path}")

    print(f"Full report: {json_path}")


if __name__ == "__main__":
    main()
