# JSON and Markdown reports for evaluation results

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from .common import ensure_dir


def format_metric(value):
    """ Format a metric value for the Markdown report """
    return "-" if value is None else f"{value:.4f}"


def _stat(values, name):
    """ Return a scalar statistic from a numeric array or None """
    if values is None or len(values) == 0:
        return None
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "max": float(values.max()),
    }[name]


def gaussian_statistics(path):
    """ Read compact size and opacity statistics from a Gaussian PLY """
    path = Path(path)
    if not path.exists():
        return {"gaussian_count": 0, "file_path": str(path)}
    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names or ())
    scales = [
        np.asarray(vertex[name], dtype=np.float64)
        for name in ("scale_0", "scale_1", "scale_2")
        if name in names
    ]
    size = np.linalg.norm(np.column_stack(scales), axis=1) if len(scales) == 3 else None
    opacity = np.asarray(vertex["opacity"], dtype=np.float64) if "opacity" in names else None
    return {
        "gaussian_count": int(len(vertex)),
        "size_min": _stat(size, "min"),
        "size_mean": _stat(size, "mean"),
        "size_std": _stat(size, "std"),
        "size_max": _stat(size, "max"),
        "opacity_min": _stat(opacity, "min"),
        "opacity_mean": _stat(opacity, "mean"),
        "opacity_std": _stat(opacity, "std"),
        "opacity_max": _stat(opacity, "max"),
        "file_path": str(path),
    }


def write_result(results_dir, result):
    """ Write one source result as JSON and a compact Markdown report """
    ensure_dir(results_dir)
    tag = result["mask_source"] # Either "yolo" or "gt2d"
    json_path = results_dir / f"results_{tag}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))

    lines = [
        f"# {result['dataset']} {result['scene']} ({tag})",
        "",
        "## Metrics by beta",
    ]

    for beta, beta_metrics in result["metrics_by_beta"].items():
        lines += [
            "",
            f"### Beta {beta}",
            f"mIoU: {format_metric(beta_metrics['mIoU'])}",
            f"GT transfer mIoU: {format_metric(beta_metrics['ground_truth_transfer_mIoU'])}",
            f"relative mIoU: {format_metric(beta_metrics['relative_mIoU'])}",
            f"global IoU: {format_metric(beta_metrics['global_iou'])}",
            f"macro precision: {format_metric(beta_metrics['macro_precision'])}",
            f"macro recall: {format_metric(beta_metrics['macro_recall'])}",
            f"global precision: {format_metric(beta_metrics['global_precision'])}",
            f"global recall: {format_metric(beta_metrics['global_recall'])}",
        ]

    lines += ["", "## Per class and beta"]

    # Add one row for every evaluated class and beta in the complete sweep
    for name, item in result["per_class"].items():
        for beta, sweep_item in item["sweep"].items():
            prediction = sweep_item["iou"]
            ground_truth_transfer = sweep_item["ground_truth_transfer_iou"]
            lines += [
                "",
                f"### {name}, beta {beta}",
                f"IoU: {format_metric(prediction['iou'])}",
                f"Precision: {format_metric(prediction['precision'])}",
                f"Recall: {format_metric(prediction['recall'])}",
                f"GT transfer IoU: {format_metric(ground_truth_transfer['iou'])}",
                f"GT transfer precision: {format_metric(ground_truth_transfer['precision'])}",
                f"GT transfer recall: {format_metric(ground_truth_transfer['recall'])}",
            ]

    # Write the Markdown report to a file named after the mask source
    (results_dir / f"results_{tag}.md").write_text("\n".join(lines) + "\n")
