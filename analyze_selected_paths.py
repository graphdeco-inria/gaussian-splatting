#!/usr/bin/env python3
"""
Analyze path overlap across selected datasets.

The script scans `selected_*` datasets (or user-provided directories) under the
gaussian_splatting data folder, counts planned paths per scene, and estimates
how many path variants are shared across datasets based on start/end proximity.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
import itertools
import sys


Point = Tuple[float, float]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_DATASETS = ["selected_33w", "selected_65k"]
DEFAULT_PATH_SAMPLES = 20


@dataclass
class PathEntry:
    dataset: str
    scene: str
    path_id: str
    start: Point
    goal: Point
    source: Path
    samples: List[Point]


@dataclass
class PathCluster:
    representative_start: Point
    representative_goal: Point
    representative_samples: List[Point]
    members: DefaultDict[str, List[PathEntry]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def matches(
        self,
        entry: PathEntry,
        endpoint_threshold: float,
        path_threshold: float,
    ) -> bool:
        if (
            euclidean(self.representative_start, entry.start) > endpoint_threshold
            or euclidean(self.representative_goal, entry.goal) > endpoint_threshold
        ):
            return False
        avg_distance = average_point_distance(
            self.representative_samples, entry.samples
        )
        return avg_distance <= path_threshold

    def add(self, entry: PathEntry) -> None:
        self.members[entry.dataset].append(entry)


def euclidean(p1: Point, p2: Point) -> float:
    return math.dist(p1, p2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze overlap between selected path datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Base directory containing the selected_* folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Specific dataset directories to inspect (default: selected_33w selected_65k).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Maximum distance (meters) between starts and goals to treat paths as identical.",
    )
    parser.add_argument(
        "--path-threshold",
        type=float,
        default=0.2,
        help="Average deviation (meters) allowed between resampled paths.",
    )
    parser.add_argument(
        "--path-samples",
        type=int,
        default=DEFAULT_PATH_SAMPLES,
        help="Number of evenly spaced samples per path for shape comparison.",
    )
    parser.add_argument(
        "--show-scenes",
        action="store_true",
        help="Print per-scene summaries instead of only the aggregated totals.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for skipped/invalid paths.",
    )
    return parser.parse_args()


def extract_point(node: Dict) -> Point | None:
    world = node.get("world") if isinstance(node, dict) else None
    if not isinstance(world, dict):
        return None
    x = world.get("x")
    y = world.get("y")
    if x is None or y is None:
        return None
    return (float(x), float(y))


def extract_path_points(path_node: Dict) -> List[Point]:
    points: List[Point] = []
    if not isinstance(path_node, dict):
        return points
    keypoints = path_node.get("keypoints_world")
    if not isinstance(keypoints, list):
        return points
    for item in keypoints:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        points.append((float(x), float(y)))
    return points


def resample_polyline(points: Sequence[Point], sample_count: int) -> List[Point]:
    if sample_count <= 0 or not points:
        return []
    if sample_count == 1 or len(points) == 1:
        return [points[0]]

    cumulative = [0.0]
    for idx in range(1, len(points)):
        cumulative.append(cumulative[-1] + euclidean(points[idx - 1], points[idx]))
    total_length = cumulative[-1]
    if total_length == 0:
        return [points[0] for _ in range(sample_count)]

    samples: List[Point] = []
    targets = [
        i * total_length / (sample_count - 1) for i in range(sample_count - 1)
    ]
    targets.append(total_length)
    seg_idx = 0
    for target in targets:
        if target >= total_length:
            samples.append(points[-1])
            continue
        while (
            seg_idx < len(cumulative) - 1 and cumulative[seg_idx + 1] < target
        ):
            seg_idx += 1
        seg_start = cumulative[seg_idx]
        seg_end = cumulative[seg_idx + 1]
        if seg_end == seg_start:
            samples.append(points[seg_idx + 1])
            continue
        ratio = (target - seg_start) / (seg_end - seg_start)
        samples.append(interpolate(points[seg_idx], points[seg_idx + 1], ratio))
    return samples


def interpolate(p1: Point, p2: Point, ratio: float) -> Point:
    return (p1[0] + (p2[0] - p1[0]) * ratio, p1[1] + (p2[1] - p1[1]) * ratio)


def average_point_distance(seq1: Sequence[Point], seq2: Sequence[Point]) -> float:
    if not seq1 and not seq2:
        return 0.0
    if not seq1 or not seq2:
        return float("inf")
    count = min(len(seq1), len(seq2))
    total = 0.0
    for idx in range(count):
        total += euclidean(seq1[idx], seq2[idx])
    return total / count


def collect_scene_paths(
    base_dir: Path,
    datasets: Sequence[str],
    sample_count: int,
) -> Tuple[Dict[str, Dict[str, List[PathEntry]]], Dict[str, Dict[str, int]]]:
    scene_map: Dict[str, Dict[str, List[PathEntry]]] = {}
    dataset_stats: Dict[str, Dict[str, int]] = {}
    for dataset in datasets:
        dataset_path = base_dir / dataset
        stats = {"path_count": 0, "scene_count": 0}
        dataset_stats[dataset] = stats
        if not dataset_path.is_dir():
            logging.warning("Dataset directory %s is missing, skipping.", dataset_path)
            continue
        scenes_with_paths = set()
        for scene_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
            scene_name = scene_dir.name
            entries: List[PathEntry] = []
            for json_file in sorted(scene_dir.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text())
                except json.JSONDecodeError as exc:
                    logging.warning("Failed to parse %s: %s", json_file, exc)
                    continue
                start_point = extract_point(data.get("start", {}))
                goal_point = extract_point(data.get("goal", {}))
                if not start_point or not goal_point:
                    logging.debug("Skipping %s missing start/goal world coords.", json_file)
                    continue
                path_points = extract_path_points(data.get("path", {}))
                if not path_points:
                    path_points = [start_point, goal_point]
                samples = resample_polyline(path_points, sample_count)
                if not samples:
                    samples = [start_point, goal_point]
                entry = PathEntry(
                    dataset=dataset,
                    scene=scene_name,
                    path_id=json_file.stem,
                    start=start_point,
                    goal=goal_point,
                    source=json_file,
                    samples=samples,
                )
                entries.append(entry)
            if not entries:
                continue
            stats["path_count"] += len(entries)
            scenes_with_paths.add(scene_name)
            scene_dataset_map = scene_map.setdefault(scene_name, {})
            scene_dataset_map.setdefault(dataset, []).extend(entries)
        stats["scene_count"] = len(scenes_with_paths)
    return scene_map, dataset_stats


def cluster_scene_paths(
    paths: Iterable[PathEntry],
    endpoint_threshold: float,
    path_threshold: float,
) -> List[PathCluster]:
    clusters: List[PathCluster] = []
    for entry in paths:
        for cluster in clusters:
            if cluster.matches(entry, endpoint_threshold, path_threshold):
                cluster.add(entry)
                break
        else:
            new_cluster = PathCluster(entry.start, entry.goal, entry.samples.copy())
            new_cluster.add(entry)
            clusters.append(new_cluster)
    return clusters


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"Data directory {data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    dataset_names = list(args.datasets)
    if len(dataset_names) == 1 and dataset_names[0].lower() in {"all", "auto"}:
        dataset_names = sorted(
            p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("selected_")
        )
    if not dataset_names:
        print("No dataset directories found. Provide --datasets or create selected_* folders.", file=sys.stderr)
        sys.exit(1)

    scene_map, dataset_stats = collect_scene_paths(
        data_dir, dataset_names, args.path_samples
    )
    dataset_pairs = list(itertools.combinations(dataset_names, 2))
    shared_counts: Dict[Tuple[str, str], int] = {pair: 0 for pair in dataset_pairs}
    union_counts: Dict[Tuple[str, str], int] = {pair: 0 for pair in dataset_pairs}
    scenes_in_common: Dict[Tuple[str, str], int] = {pair: 0 for pair in dataset_pairs}

    unique_paths_total = 0

    if args.show_scenes:
        print("Per-scene summary:\n")

    for scene_name in sorted(scene_map):
        dataset_entries = scene_map[scene_name]
        scene_paths = [entry for entries in dataset_entries.values() for entry in entries]
        clusters = cluster_scene_paths(
            scene_paths, args.threshold, args.path_threshold
        )
        unique_paths_total += len(clusters)

        if dataset_pairs:
            for pair in dataset_pairs:
                first_entries = dataset_entries.get(pair[0])
                second_entries = dataset_entries.get(pair[1])
                if not first_entries or not second_entries:
                    continue
                scenes_in_common[pair] += 1
                for cluster in clusters:
                    present = {
                        dataset for dataset, entries in cluster.members.items() if entries
                    }
                    union_counts[pair] += 1
                    if pair[0] in present and pair[1] in present:
                        shared_counts[pair] += 1

        if args.show_scenes:
            line_parts = [f"{scene_name}: unique_paths={len(clusters)}"]
            for dataset in dataset_names:
                count = len(dataset_entries.get(dataset, []))
                line_parts.append(f"{dataset} paths={count}")
            print(", ".join(line_parts))

    print("\nDataset totals:")
    for dataset in dataset_names:
        stats = dataset_stats.get(dataset, {"scene_count": 0, "path_count": 0})
        print(
            f"- {dataset}: scenes_with_paths={stats['scene_count']}, "
            f"path_count={stats['path_count']}"
        )

    if dataset_pairs:
        print("\nPath overlap between datasets:")
        for pair in dataset_pairs:
            shared = shared_counts[pair]
            union = union_counts[pair]
            percent = (shared / union * 100.0) if union else 0.0
            common_scenes = scenes_in_common[pair]
            print(
                f"- {pair[0]} vs {pair[1]}: shared_path_variants={shared}, "
                f"union_variants={union}, percentage_shared={percent:.2f}%, "
                f"scenes_in_common={common_scenes}"
            )

    print(f"\nTotal unique path variants across datasets: {unique_paths_total}")


if __name__ == "__main__":
    main()
