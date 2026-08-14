# CSV tables that are only appended and used for the analysis of the validation scenes

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import safe_name
from .reporting import gaussian_statistics


# Each relation is stored in its own CSV so individual analyses can be extended without rewriting the records produced by the other stages
SCHEMA = {
    "runs": [
        "run_id", "created_at", "status", "dataset", "scene_id",
        "scene_name", "split", "source", "output_root", "model_root",
    ],

    "run_parameters": [
        "run_id", "evaluation_scope_version", "dataset", "scene", "split",
        "data_root", "iterations", "resolution", "train_data_device",
        "vote_data_device", "sequence_name", "frame_step", "yolo_conf",
        "size_measure", "hysteresis_gamma", "hysteresis_radius",
        "background_mode", "background_confidence", "background_view_policy",
        "betas",
        "tau", "min_share", "mesh_to_gaussian_transfer",
        "gaussian_to_mesh_transfer", "min_opacity",
        "gaussian_to_mesh_background_competes", "mesh_to_gaussian_background_competes",
        "opacity_weighting", "sigma", "size_penalty",
        "raster_block_size", "mask_source",
    ],

    "scenes": [
        "scene_id", "dataset", "scene_name", "split", "scene_path",
        "num_vertices", "num_images",
    ],

    "camera_statistics": [
        "scene_id", "dataset", "num_images",
        "width_min", "width_mean", "width_max",
        "height_min", "height_mean", "height_max",
        "fx_min", "fx_mean", "fx_max", "fy_min", "fy_mean", "fy_max",
        "cx_min", "cx_mean", "cx_max", "cy_min", "cy_mean", "cy_max",
    ],

    "classes": ["class_id", "dataset", "class_name", "detector_name", "detector_stored_id"],

    "run_sources": ["run_id", "source", "mask_directory", "segmentation_directory"],

    "scene_classes": [
        "scene_id", "class_id", "gt_vertex_count", "gt_visible_vertex_count",
        "gt_evaluated_vertex_count",
    ],

    "run_betas": ["run_id", "source", "beta_id", "beta_order", "beta"],
    "vote_statistics": [
        "run_id", "source", "class_id", "num_cameras", "num_class_views",
        "num_gaussians", "target_weight_sum", "background_weight_sum",
        "supported_gaussians", "target_score_mean", "target_score_std",
        "target_score_min", "target_score_p05", "target_score_p25",
        "target_score_median", "target_score_p75", "target_score_p90",
        "target_score_p92_5", "target_score_p95", "target_score_p97_5",
        "target_score_p99", "target_score_p99_9",
        "target_score_max", "supported_fraction",
    ],

    "gaussian_statistics": [
        "run_id", "source", "class_id", "beta_id", "set_type", "gaussian_count",
        "size_min", "size_mean", "size_std", "size_max", "opacity_min",
        "opacity_mean", "opacity_std", "opacity_max", "file_path",
    ],

    "class_beta_metrics": [
        "run_id", "source", "class_id", "beta_id", "beta", "tp", "fp", "fn",
        "gt_count", "pred_count", "precision", "recall", "iou",
        "ground_truth_transfer_tp", "ground_truth_transfer_fp",
        "ground_truth_transfer_fn", "ground_truth_transfer_gt_count",
        "ground_truth_transfer_pred_count",
        "ground_truth_transfer_precision", "ground_truth_transfer_recall",
        "ground_truth_transfer_iou", "relative_iou",
    ],

    "aggregate_beta_metrics": [
        "run_id", "source", "beta_id", "beta", "mIoU", "global_iou",
        "macro_precision", "macro_recall", "global_precision", "global_recall",
        "ground_truth_transfer_mIoU", "ground_truth_transfer_macro_precision",
        "ground_truth_transfer_macro_recall",
        "ground_truth_transfer_global_precision",
        "ground_truth_transfer_global_recall", "relative_mIoU",
        "evaluated_classes", "relative_classes",
    ],

}


def utc_now():
    """ Return a UTC timestamp for CSV records """
    return datetime.now(timezone.utc).isoformat()


def _summary(values):
    """Return compact descriptive statistics for a numeric camera field."""
    values = [float(value) for value in values]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


class AnalyticsStore:
    """ Write one CSV file for only appends per analytical relation """

    def __init__(self, root):
        """ Create the analytics directory and initialize every table header once """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        # Initialize every table with its header if the file does not exist or is empty
        for table, fields in SCHEMA.items():
            path = self.root / f"{table}.csv"
            if not path.exists() or path.stat().st_size == 0:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fields).writeheader()

    def append(self, table, row):
        """ Append one row while preserving the table schema and previous rows """
        fields = SCHEMA[table]

        # Ignore extra keys and fill missing fields with empty CSV values
        values = {field: row.get(field) for field in fields}
        with (self.root / f"{table}.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(values)

    def append_unique(self, table, row, key_fields):
        """ Append a catalog row only when its key is not already present """
        path = self.root / f"{table}.csv"

        # Compare string representations because CSV values are read as strings
        key = tuple(str(row.get(field, "")) for field in key_fields)
        with path.open("r", newline="", encoding="utf-8") as handle:
            for existing in csv.DictReader(handle):
                if tuple(existing.get(field, "") for field in key_fields) == key:
                    return
        self.append(table, row)


def record_scene_analytics(store, args, scene, scene_id):
    """Record the scene and its class level ground truth support once."""
    evaluation_mask = scene.evaluation_mask
    store.append_unique("scenes", {
        "scene_id": scene_id,
        "dataset": scene.dataset,
        "scene_name": scene.scene,
        "split": args.split,
        "scene_path": str(getattr(scene, "scene_root", "")),
        "num_vertices": len(scene.vertices),
        "num_images": scene.num_images,
    }, ["scene_id"])

    camera_fields = {
        "width": [row["width"] for row in scene.camera_intrinsics],
        "height": [row["height"] for row in scene.camera_intrinsics],
        "fx": [row["fx"] for row in scene.camera_intrinsics],
        "fy": [row["fy"] for row in scene.camera_intrinsics],
        "cx": [row["cx"] for row in scene.camera_intrinsics],
        "cy": [row["cy"] for row in scene.camera_intrinsics],
    }
    camera_row = {
        "scene_id": scene_id,
        "dataset": scene.dataset,
        "num_images": scene.num_images,
    }
    for field, values in camera_fields.items():
        summary = _summary(values)
        camera_row.update({
            f"{field}_min": summary["min"],
            f"{field}_mean": summary["mean"],
            f"{field}_max": summary["max"],
        })
    store.append_unique("camera_statistics", camera_row, ["scene_id"])
    # ``class_id`` is the SceneData local main ID, not a detector mask ID or
    # a source dataset ID.
    for class_id, spec in enumerate(scene.classes):
        class_mask = scene.semantic_labels == class_id
        store.append_unique("classes", {
            "class_id": f"{scene.dataset}:{class_id}",
            "dataset": scene.dataset,
            "class_name": spec.name,
            "detector_name": spec.name_by_detector,
            "detector_stored_id": spec.detector_stored_id,
        }, ["class_id"])
        store.append_unique("scene_classes", {
            "scene_id": scene_id,
            "class_id": f"{scene.dataset}:{class_id}",
            "gt_vertex_count": int(class_mask.sum()),
            "gt_visible_vertex_count": int((class_mask & scene.visible).sum()),
            "gt_evaluated_vertex_count": int((class_mask & evaluation_mask).sum()),
        }, ["scene_id", "class_id"])


def record_source_analytics(store, run_id, source, scene, classes, betas,
                            source_dir, result, model_ply):
    """Record votes, Gaussian summaries and metrics for one mask source."""
    store.append("gaussian_statistics", {
        "run_id": run_id,
        "source": source,
        "set_type": "full_model",
        **gaussian_statistics(model_ply),
    })
    for beta_order, beta in enumerate(betas, start=1):
        store.append("run_betas", {
            "run_id": run_id,
            "source": source,
            "beta_id": f"{run_id}:{source}:{beta_order}",
            "beta_order": beta_order,
            "beta": beta,
        })

    for spec in classes:
        class_id = scene.class_id(spec.name) # SceneData local main ID.
        analytics_class_id = f"{scene.dataset}:{class_id}"
        safe = safe_name(spec.name_by_detector)
        class_dir = source_dir / safe
        vote_stats_path = class_dir / "vote_statistics.json"
        if vote_stats_path.exists():
            vote_stats = json.loads(vote_stats_path.read_text())
            vote_stats.update({
                "run_id": run_id,
                "source": source,
                "class_id": analytics_class_id,
            })
            store.append("vote_statistics", vote_stats)

        ground_truth_transfer_path = class_dir / "ground_truth_gaussians.ply"
        ground_truth_transfer_stats = gaussian_statistics(ground_truth_transfer_path)
        store.append("gaussian_statistics", {
            "run_id": run_id,
            "source": source,
            "class_id": analytics_class_id,
            "set_type": "ground_truth_transfer",
            **ground_truth_transfer_stats,
        })
        item = result["per_class"].get(spec.name, {})
        for beta_order, beta in enumerate(betas, start=1):
            beta_id = f"{run_id}:{source}:{beta_order}"
            beta_key = str(beta)
            sweep = item.get("sweep", {}).get(beta_key)
            predicted_path = class_dir / (
                f"labeled_gaussians_{safe}_beta{str(beta).replace('.', '_')}.ply"
            )
            stats = gaussian_statistics(predicted_path)
            store.append("gaussian_statistics", {
                "run_id": run_id,
                "source": source,
                "class_id": analytics_class_id,
                "beta_id": beta_id,
                "set_type": "predicted",
                **stats,
            })
            if sweep is None:
                continue
            prediction = sweep["iou"]
            ground_truth_transfer_metrics = sweep["ground_truth_transfer_iou"]
            store.append("class_beta_metrics", {
                "run_id": run_id,
                "source": source,
                "class_id": analytics_class_id,
                "beta_id": beta_id,
                "beta": beta,
                "tp": prediction["tp"],
                "fp": prediction["fp"],
                "fn": prediction["fn"],
                "gt_count": prediction["gt_count"],
                "pred_count": prediction["pred_count"],
                "precision": prediction["precision"],
                "recall": prediction["recall"],
                "iou": prediction["iou"],
                "ground_truth_transfer_tp": ground_truth_transfer_metrics["tp"],
                "ground_truth_transfer_fp": ground_truth_transfer_metrics["fp"],
                "ground_truth_transfer_fn": ground_truth_transfer_metrics["fn"],
                "ground_truth_transfer_gt_count": ground_truth_transfer_metrics["gt_count"],
                "ground_truth_transfer_pred_count": ground_truth_transfer_metrics["pred_count"],
                "ground_truth_transfer_precision": ground_truth_transfer_metrics["precision"],
                "ground_truth_transfer_recall": ground_truth_transfer_metrics["recall"],
                "ground_truth_transfer_iou": ground_truth_transfer_metrics["iou"],
                "relative_iou": sweep["relative_iou"],
            })

    for beta_order, beta in enumerate(betas, start=1):
        aggregate = result["metrics_by_beta"].get(str(beta))
        if aggregate is None:
            continue
        store.append("aggregate_beta_metrics", {
            "run_id": run_id,
            "source": source,
            "beta_id": f"{run_id}:{source}:{beta_order}",
            "beta": beta,
            **aggregate,
        })
