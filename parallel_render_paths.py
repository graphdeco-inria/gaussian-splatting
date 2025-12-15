#!/usr/bin/env python3
"""
Plan and run render_label_paths.py jobs in parallel while respecting per-path actor assignments.

Typical workflow:
  1. Generate actor stats (optional) via analyze_actor_sequences.py.
  2. Produce a pairing manifest with random_actor_assignments.py (stores seed + actors per label).
  3. Use this script to filter label paths, shard work across workers, and launch render_label_paths.py
     once per (scene, actor) group. Outputs per-job logs and a final JSON report of all pairings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from argparse import BooleanOptionalAction

from render_label_paths import (  # type: ignore
    DEFAULT_ACTOR_PATTERN,
    DEFAULT_ACTOR_SPEED,
    DEFAULT_VIDEO_FPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ERROR_LOG,
    SCENES_DIR,
    TASK_OUTPUT_DIR,
    load_occupancy_metadata,
    prepare_path_data,
    estimate_actor_frame_count,
    resolve_label_directory,
    PathSampler,
)


@dataclass
class AssignmentEntry:
    scene: str
    label: str
    actor_id: str
    actor_dir: Path
    actor_pattern: str
    actor_height: float
    actor_speed: float
    actor_fps: float
    follow_distance: float
    follow_buffer: float
    actor_foot_offset: float
    animation_cycle_mod: int
    actor_loop: bool


@dataclass
class LabelTask:
    assignment: AssignmentEntry
    json_path: Path
    path_length: float
    estimated_frames: int


@dataclass
class JobPlan:
    scene: str
    actor_id: str
    labels: list[str]
    assignment: AssignmentEntry


class ProgressTracker:
    """Maintain aggregated runtime stats and write progress JSON after each job."""

    STAGE_KEYS = (
        "mp4_write_sec",
        "perframe_png_sec",
        "perframe_depth_sec",
        "ply_write_sec",
    )

    def __init__(self, *, total_jobs: int, total_paths: int, output_path: Path | None):
        self.total_jobs = total_jobs
        self.total_paths = total_paths
        self.completed_jobs = 0
        self.completed_paths = 0
        self.total_render_time = 0.0
        self.total_frames_rendered = 0
        self.stage_totals: defaultdict[str, float] = defaultdict(float)
        self.slot_vram: defaultdict[str, list[float]] = defaultdict(list)
        self.paths: list[dict] = []
        self.output_path = output_path.resolve() if output_path is not None else None
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_summary(None)

    def record_job(self, job_result: dict, job_metrics: dict | None) -> None:
        self.completed_jobs += 1
        if job_metrics:
            for entry in job_metrics.get("paths", []) or []:
                self.paths.append(entry)
                self.completed_paths += 1
                try:
                    duration = float(entry.get("duration_sec") or 0.0)
                except (TypeError, ValueError):
                    duration = 0.0
                self.total_render_time += duration
                try:
                    frames_rendered = int(entry.get("frames") or 0)
                except (TypeError, ValueError):
                    frames_rendered = 0
                self.total_frames_rendered += frames_rendered
                stage_seconds = entry.get("stage_seconds") or {}
                for stage, seconds in stage_seconds.items():
                    try:
                        self.stage_totals[stage] += float(seconds)
                    except (TypeError, ValueError):
                        continue
                slot = entry.get("job_slot")
                avg_vram = entry.get("vram_avg_bytes")
                if slot is not None and isinstance(avg_vram, (int, float)):
                    self.slot_vram[str(slot)].append(float(avg_vram))
        self._write_summary(job_result)

    def _write_summary(self, last_job: dict | None) -> None:
        if self.output_path is None:
            return
        if self.total_paths <= 0:
            progress_pct = 1.0
            remaining_paths = 0
        else:
            progress_pct = self.completed_paths / self.total_paths
            remaining_paths = max(self.total_paths - self.completed_paths, 0)
        avg_path_time = (
            self.total_render_time / self.completed_paths
            if self.completed_paths > 0
            else None
        )
        avg_frame_time = (
            self.total_render_time / self.total_frames_rendered
            if self.total_frames_rendered > 0
            else None
        )
        avg_frames_per_path = (
            self.total_frames_rendered / self.completed_paths
            if self.completed_paths > 0
            else None
        )
        eta_seconds = None
        if avg_path_time is not None and remaining_paths > 0:
            eta_seconds = avg_path_time * remaining_paths

        stage_total = sum(self.stage_totals.get(stage, 0.0) for stage in self.STAGE_KEYS)
        stage_ratios = {}
        for stage in self.STAGE_KEYS:
            value = self.stage_totals.get(stage, 0.0)
            stage_ratios[stage] = (value / stage_total) if stage_total > 0 else 0.0

        avg_vram_per_thread = {
            slot: (sum(values) / len(values))
            for slot, values in self.slot_vram.items()
            if values
        }

        summary = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "total_paths": self.total_paths,
            "completed_paths": self.completed_paths,
            "progress_pct": progress_pct,
            "avg_path_time_sec": avg_path_time,
            "avg_frame_time_sec": avg_frame_time,
            "avg_frames_per_path": avg_frames_per_path,
            "eta_seconds": eta_seconds,
            "avg_vram_per_thread": avg_vram_per_thread,
            "stage_ratio": stage_ratios,
        }
        if last_job is not None:
            summary["last_job"] = {
                "scene": last_job.get("scene"),
                "actor_id": last_job.get("actor_id"),
                "status": last_job.get("status"),
                "duration_sec": last_job.get("duration_sec"),
            }

        payload = {
            "summary": summary,
            "paths": self.paths,
        }
        self.output_path.write_text(json.dumps(payload, indent=2))


def _load_metrics_json(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[METRICS] WARN: Metrics file missing at {path}; job may have failed early.", flush=True)
        return None
    stripped = text.strip()
    if not stripped:
        print(f"[METRICS] WARN: Metrics file {path} is empty; skipping entry.", flush=True)
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        print(f"[METRICS] WARN: Could not parse metrics {path}: {exc}", flush=True)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shard render_label_paths.py invocations per actor/scene pairing."
    )
    parser.add_argument(
        "--assignment-manifest",
        type=Path,
        required=True,
        help="JSON manifest produced by random_actor_assignments.py.",
    )
    parser.add_argument(
        "--render-script",
        type=Path,
        default=Path(__file__).with_name("render_label_paths.py"),
        help="Path to the render_label_paths.py script (default: alongside this file).",
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
        "--workers",
        type=int,
        default=1,
        help="Number of parallel render_label_paths.py processes to launch (default: 1).",
    )
    parser.add_argument(
        "--minimal-frames",
        type=int,
        default=None,
        help="Pre-filter label paths whose estimated frame count falls below this threshold.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride used while sampling raster_world waypoints (default: 1).",
    )
    parser.add_argument(
        "--swap-xy",
        action="store_true",
        help="Match render_label_paths.py --swap-xy during pre-filtering.",
    )
    parser.add_argument(
        "--mirror-translation",
        action=BooleanOptionalAction,
        default=True,
        help="Mirror raster_world coordinates like render_label_paths.py (default: True).",
    )
    parser.add_argument(
        "--render-extra-args",
        action="append",
        default=[],
        help="Additional CLI snippet appended to every render_label_paths.py command "
        "(example: --render-extra-args \"--overwrite --gpu-only\").",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root for render_label_paths.py (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERROR_LOG,
        help=f"Per-process error log path (default: {DEFAULT_ERROR_LOG}).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("parallel_render_logs"),
        help="Directory where per-job stdout/stderr logs will be written.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("parallel_render_report.json"),
        help="Destination JSON report detailing all jobs and path assignments.",
    )
    parser.add_argument(
        "--per-job-metrics-dir",
        type=Path,
        default=Path("parallel_metrics"),
        help="Directory where per-job metrics JSON files will be written (default: parallel_metrics).",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=Path("datagen_progress.json"),
        help="Aggregated progress JSON path updated after each job (default: datagen_progress.json).",
    )
    parser.add_argument(
        "--scene-shard-index",
        type=int,
        default=None,
        help="1-based shard index for splitting scenes across multiple runs (requires --scene-shard-count).",
    )
    parser.add_argument(
        "--scene-shard-count",
        type=int,
        default=None,
        help="Total number of scene shards. Scenes are sorted alphabetically and distributed round-robin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan jobs and write the report but skip spawning render_label_paths.py.",
    )
    parser.add_argument(
        "--skip-completed-log",
        type=Path,
        default=None,
        help="Optional log whose [SUCCESS] entries list scene/actor pairs that have already completed.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[dict[str, dict], list[AssignmentEntry], dict[str, Any]]:
    manifest = json.loads(path.read_text())
    actor_map: dict[str, dict] = {}
    for entry in manifest.get("actors", []):
        actor_map[entry["id"]] = entry
    assignments: list[AssignmentEntry] = []
    for row in manifest.get("assignments", []):
        actor_id = row["actor_id"]
        actor_info = actor_map.get(actor_id)
        if not actor_info:
            raise KeyError(f"Actor {actor_id} referenced by assignment but missing from manifest.")
        assignments.append(
            AssignmentEntry(
                scene=row["scene"],
                label=row["label"],
                actor_id=actor_id,
                actor_dir=Path(actor_info["directory"]),
                actor_pattern=actor_info.get("pattern", DEFAULT_ACTOR_PATTERN),
                actor_height=float(actor_info.get("height", 1.7)),
                actor_speed=float(actor_info.get("speed", DEFAULT_ACTOR_SPEED)),
                actor_fps=float(actor_info.get("fps", DEFAULT_VIDEO_FPS)),
                follow_distance=float(actor_info.get("follow_distance", 1.5)),
                follow_buffer=float(actor_info.get("follow_buffer", 0.5)),
                actor_foot_offset=float(row.get("actor_foot_offset", actor_info.get("foot_offset", 0.0))),
                animation_cycle_mod=int(actor_info.get("animation_cycle_mod", 3)),
                actor_loop=bool(actor_info.get("loop", True)),
            )
        )
    return actor_map, assignments, manifest


def gather_label_tasks(
    assignments: Sequence[AssignmentEntry],
    *,
    scenes_dir: Path,
    tasks_dir: Path,
    stride: int,
    swap_xy: bool,
    mirror_translation: bool,
    minimal_frames: int | None,
) -> tuple[list[LabelTask], list[dict]]:
    cache_meta: dict[str, dict] = {}
    cache_label_dir: dict[str, Path | None] = {}
    tasks: list[LabelTask] = []
    skipped: list[dict] = []

    def get_meta(scene_id: str) -> dict | None:
        if scene_id not in cache_meta:
            dataset_dir = scenes_dir / scene_id
            if not dataset_dir.is_dir():
                cache_meta[scene_id] = None
            else:
                try:
                    cache_meta[scene_id] = load_occupancy_metadata(dataset_dir)
                except Exception:
                    cache_meta[scene_id] = None
        return cache_meta[scene_id]

    def get_label_dir(scene_id: str) -> Path | None:
        if scene_id not in cache_label_dir:
            scene_task_dir = tasks_dir / scene_id
            cache_label_dir[scene_id] = resolve_label_directory(scene_task_dir)
        return cache_label_dir[scene_id]

    for entry in assignments:
        meta = get_meta(entry.scene)
        label_dir = get_label_dir(entry.scene)
        if meta is None or label_dir is None:
            skipped.append(
                {
                    "scene": entry.scene,
                    "label": entry.label,
                    "reason": "missing_meta_or_labels",
                }
            )
            continue
        json_path = label_dir / f"{entry.label}.json"
        if not json_path.is_file():
            skipped.append({"scene": entry.scene, "label": entry.label, "reason": "label_missing"})
            continue
        try:
            prepared = prepare_path_data(
                json_path=json_path,
                meta=meta,
                stride=max(1, stride),
                mirror_translation=mirror_translation,
                swap_xy=swap_xy,
            )
            path_length = float(PathSampler(prepared.path_xy).total_length) if len(prepared.path_xy) >= 2 else 0.0
            estimated_frames = estimate_actor_frame_count(
                prepared.path_xy,
                follow_distance=entry.follow_distance,
            )
        except Exception as exc:  # pylint: disable=broad-except
            skipped.append({"scene": entry.scene, "label": entry.label, "reason": f"prepare_failed: {exc}"})
            continue

        if minimal_frames is not None and estimated_frames < minimal_frames:
            skipped.append(
                {
                    "scene": entry.scene,
                    "label": entry.label,
                    "reason": f"below_min_frames({estimated_frames}<{minimal_frames})",
                }
            )
            continue

        tasks.append(
            LabelTask(
                assignment=entry,
                json_path=json_path,
                path_length=path_length,
                estimated_frames=estimated_frames,
            )
        )

    return tasks, skipped


def build_job_plans(tasks: Sequence[LabelTask]) -> list[JobPlan]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    assignment_lookup: dict[tuple[str, str], AssignmentEntry] = {}
    for task in tasks:
        key = (task.assignment.scene, task.assignment.actor_id)
        grouped[key].append(task.assignment.label)
        assignment_lookup[key] = task.assignment

    plans: list[JobPlan] = []
    for (scene, actor_id), labels in grouped.items():
        plans.append(
            JobPlan(
                scene=scene,
                actor_id=actor_id,
                labels=sorted(labels),
                assignment=assignment_lookup[(scene, actor_id)],
            )
        )
    return plans


RESUME_STATUS_RE = re.compile(
    r"\[(?P<status>SUCCESS|FAILED)\]\s+scene=(?P<scene>\S+)\s+actor=(?P<actor>\S+)"
)


def load_resume_statuses_from_log(log_path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Resume log not found: {log_path}")
    completed: set[tuple[str, str]] = set()
    failed: set[tuple[str, str]] = set()
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = RESUME_STATUS_RE.search(raw_line)
            if not match:
                continue
            key = (match.group("scene"), match.group("actor"))
            status = match.group("status")
            if status == "SUCCESS":
                completed.add(key)
                failed.discard(key)
            else:
                if key not in completed:
                    failed.add(key)
    return completed, failed


def filter_tasks_by_scene_shard(
    tasks: Sequence[LabelTask],
    shard_index: int | None,
    shard_count: int | None,
) -> list[LabelTask]:
    if shard_index is None and shard_count is None:
        return list(tasks)
    if shard_index is None or shard_count is None:
        raise ValueError("Both --scene-shard-index and --scene-shard-count must be specified together.")
    if shard_count <= 0:
        raise ValueError("--scene-shard-count must be positive.")
    zero_based = shard_index - 1
    if zero_based < 0 or zero_based >= shard_count:
        raise ValueError("--scene-shard-index must be between 1 and --scene-shard-count inclusive.")
    scene_ids = sorted({task.assignment.scene for task in tasks})
    selected_scenes = {
        scene for idx, scene in enumerate(scene_ids) if (idx % shard_count) == zero_based
    }
    return [task for task in tasks if task.assignment.scene in selected_scenes]


def make_command(
    plan: JobPlan,
    *,
    render_script: Path,
    scenes_dir: Path,
    tasks_dir: Path,
    stride: int,
    swap_xy: bool,
    mirror_translation: bool,
    minimal_frames: int | None,
    output_dir: Path,
    error_log: Path,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(render_script),
        "--scene",
        plan.scene,
        "--scenes-dir",
        str(scenes_dir),
        "--tasks-dir",
        str(tasks_dir),
    ]
    for label in plan.labels:
        cmd.extend(["--label-id", label])
    cmd.extend(
        [
            "--actor-seq-dir",
            str(plan.assignment.actor_dir),
            "--actor-pattern",
            plan.assignment.actor_pattern or DEFAULT_ACTOR_PATTERN,
            "--actor-height",
            f"{plan.assignment.actor_height:.6f}",
            "--actor-speed",
            f"{plan.assignment.actor_speed:.6f}",
            "--actor-fps",
            f"{plan.assignment.actor_fps:.6f}",
            "--follow-distance",
            f"{plan.assignment.follow_distance:.6f}",
            "--follow-buffer",
            f"{plan.assignment.follow_buffer:.6f}",
            "--actor-foot-offset",
            f"{plan.assignment.actor_foot_offset:.6f}",
            "--animation-cycle-mod",
            str(plan.assignment.animation_cycle_mod),
            "--output-dir",
            str(output_dir),
            "--error-log",
            str(error_log),
            "--stride",
            str(stride),
        ]
    )
    if not plan.assignment.actor_loop:
        cmd.append("--actor-no-loop")
    if swap_xy:
        cmd.append("--swap-xy")
    if not mirror_translation:
        cmd.append("--no-mirror-translation")
    if minimal_frames is not None:
        cmd.extend(["--minimal-frames", str(minimal_frames)])
    cmd.extend(extra_args)
    return cmd


def _print_failure_log(log_path: Path) -> None:
    try:
        contents = log_path.read_text()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Unable to read log {log_path}: {exc}", flush=True)
        return
    header = f"[LOG][{log_path}]"
    print(f"{header} ---- BEGIN ----", flush=True)
    if contents.strip():
        print(contents.rstrip("\n"), flush=True)
    else:
        print("(log is empty)", flush=True)
    print(f"{header} ---- END ----", flush=True)


def run_job(
    idx: int,
    plan: JobPlan,
    cmd: list[str],
    *,
    log_dir: Path,
    dry_run: bool,
    job_name: str | None = None,
    metrics_path: Path | None = None,
) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = job_name or f"{idx:04d}_{plan.scene}_{plan.actor_id}"
    temp_log_path = log_dir / f"{base_name}.log"
    final_log_path = temp_log_path
    pid: int | None = None
    start = time.time()
    if dry_run:
        duration = 0.0
        return {
            "scene": plan.scene,
            "actor_id": plan.actor_id,
            "labels": plan.labels,
            "cmd": cmd,
            "log": str(temp_log_path),
            "status": "dry-run",
            "returncode": None,
            "pid": None,
            "duration_sec": duration,
            "job_name": base_name,
            "metrics_path": str(metrics_path) if metrics_path else None,
        }

    with temp_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        pid = proc.pid
        log_file.write(f"[INFO] PID={pid}\n")
        log_file.flush()
        proc.wait()
        returncode = proc.returncode
    duration = time.time() - start
    if pid is not None:
        candidate = log_dir / f"{base_name}_pid{pid}.log"
        try:
            if candidate.exists():
                candidate.unlink()
            temp_log_path.rename(candidate)
            final_log_path = candidate
        except Exception:
            final_log_path = temp_log_path
    status = "success" if returncode == 0 else "failed"
    return {
        "scene": plan.scene,
        "actor_id": plan.actor_id,
        "labels": plan.labels,
        "cmd": cmd,
        "log": str(final_log_path),
        "status": status,
        "returncode": returncode,
        "pid": pid,
        "duration_sec": duration,
        "job_name": base_name,
        "metrics_path": str(metrics_path) if metrics_path else None,
    }


def main() -> None:
    args = parse_args()
    actor_map, assignments, manifest = load_manifest(args.assignment_manifest)
    if not assignments:
        raise SystemExit("Assignment manifest does not contain any label-path pairings.")

    extra_args: list[str] = []
    for snippet in args.render_extra_args:
        extra_args.extend(shlex.split(snippet))

    tasks, skipped = gather_label_tasks(
        assignments,
        scenes_dir=args.scenes_dir,
        tasks_dir=args.tasks_dir,
        stride=args.stride,
        swap_xy=args.swap_xy,
        mirror_translation=args.mirror_translation,
        minimal_frames=args.minimal_frames,
    )
    tasks = filter_tasks_by_scene_shard(tasks, args.scene_shard_index, args.scene_shard_count)
    if not tasks:
        print("[WARN] No label paths satisfied the current filters.", flush=True)

    plans = build_job_plans(tasks)
    plans.sort(key=lambda p: (p.scene, p.actor_id))
    skipped_jobs_from_log = 0
    if args.skip_completed_log is not None:
        completed_pairs, failed_pairs = load_resume_statuses_from_log(args.skip_completed_log)
        if completed_pairs:
            before = len(plans)
            plans = [plan for plan in plans if (plan.scene, plan.actor_id) not in completed_pairs]
            skipped_jobs_from_log = before - len(plans)
            print(
                f"[RESUME] Loaded {len(completed_pairs)} completed jobs from {args.skip_completed_log}.",
                flush=True,
            )
            if skipped_jobs_from_log:
                print(f"[RESUME] Skipping {skipped_jobs_from_log} jobs listed as successful.", flush=True)
        else:
            print(f"[RESUME] No [SUCCESS] entries found in {args.skip_completed_log}.", flush=True)
        if failed_pairs:
            print(
                f"[RESUME] {len(failed_pairs)} jobs recorded as FAILED in the log will be retried.",
                flush=True,
            )
    print(f"[PLAN] {len(plans)} jobs will cover {len(tasks)} label paths (skipped {len(skipped)}).", flush=True)

    progress_tracker = ProgressTracker(
        total_jobs=len(plans),
        total_paths=len(tasks),
        output_path=args.progress_json,
    )
    max_workers = max(1, args.workers)

    results: list[dict] = []
    if args.dry_run or not plans:
        for idx, plan in enumerate(plans, start=1):
            cmd = make_command(
                plan,
                render_script=args.render_script,
                scenes_dir=args.scenes_dir,
                tasks_dir=args.tasks_dir,
                stride=args.stride,
                swap_xy=args.swap_xy,
                mirror_translation=args.mirror_translation,
                minimal_frames=args.minimal_frames,
                output_dir=args.output_dir,
                error_log=args.error_log,
                extra_args=extra_args,
            )
            job_name = f"{idx:04d}_{plan.scene}_{plan.actor_id}"
            job_slot = ((idx - 1) % max_workers) + 1
            cmd.extend(["--job-slot", str(job_slot), "--job-name", job_name, "--job-actor-id", plan.actor_id])
            result = run_job(
                idx,
                plan,
                cmd,
                log_dir=args.log_dir,
                dry_run=True,
                job_name=job_name,
            )
            results.append(result)
            progress_tracker.record_job(result, None)
    else:
        metrics_dir = args.per_job_metrics_dir.resolve()
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx: dict[Any, int] = {}
            for idx, plan in enumerate(plans, start=1):
                cmd = make_command(
                    plan,
                    render_script=args.render_script,
                    scenes_dir=args.scenes_dir,
                    tasks_dir=args.tasks_dir,
                    stride=args.stride,
                    swap_xy=args.swap_xy,
                    mirror_translation=args.mirror_translation,
                    minimal_frames=args.minimal_frames,
                    output_dir=args.output_dir,
                    error_log=args.error_log,
                    extra_args=extra_args,
                )
                job_name = f"{idx:04d}_{plan.scene}_{plan.actor_id}"
                job_slot = ((idx - 1) % max_workers) + 1
                cmd.extend(["--job-slot", str(job_slot), "--job-name", job_name, "--job-actor-id", plan.actor_id])
                metrics_path = metrics_dir / f"{job_name}.json"
                cmd.extend(["--metrics-json", str(metrics_path)])
                future = pool.submit(
                    run_job,
                    idx,
                    plan,
                    cmd,
                    log_dir=args.log_dir,
                    dry_run=False,
                    job_name=job_name,
                    metrics_path=metrics_path,
                )
                future_to_idx[future] = idx
            for future in as_completed(future_to_idx):
                result = future.result()
                results.append(result)
                print(
                    f"[{result['status'].upper()}] scene={result['scene']} actor={result['actor_id']} "
                    f"labels={len(result['labels'])} returncode={result['returncode']} "
                    f"pid={result.get('pid')}",
                    flush=True,
                )
                if result["status"] == "failed":
                    _print_failure_log(Path(result["log"]))
                metrics_payload = None
                metrics_file = result.get("metrics_path")
                if metrics_file:
                    metrics_payload = _load_metrics_json(Path(metrics_file))
                progress_tracker.record_job(result, metrics_payload)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assignment_manifest": str(args.assignment_manifest.resolve()),
        "seed": manifest.get("seed"),
        "jobs": results,
        "skipped_labels": skipped,
        "total_jobs": len(plans),
        "executed_jobs": sum(1 for r in results if r["status"] not in ("dry-run",)),
        "successful_jobs": sum(1 for r in results if r["status"] == "success"),
        "failed_jobs": sum(1 for r in results if r["status"] == "failed"),
        "skip_completed_log": str(args.skip_completed_log) if args.skip_completed_log else None,
        "jobs_skipped_via_log": skipped_jobs_from_log,
        "path_assignments": [
            {
                "scene": task.assignment.scene,
                "label": task.assignment.label,
                "actor_id": task.assignment.actor_id,
                "actor_dir": str(task.assignment.actor_dir),
                "path_length_m": task.path_length,
                "estimated_frames": task.estimated_frames,
            }
            for task in tasks
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2))
    print(f"[DONE] Report saved to {args.report_out}")


if __name__ == "__main__":
    main()
