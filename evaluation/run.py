# The workflow that evaluates both Scannet++ and Replica datasets

import argparse
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import cache, ground_truth, metrics, reporting, transfer
from .analytics import AnalyticsStore, record_scene_analytics, record_source_analytics, utc_now
from .common import safe_name, target_classes_by_detector
from .runtime import Runtime

from .replica.scene import ReplicaScene
from .scannetpp.scene import ScannetScene


DEFAULT_DATA_ROOT = Path("/mnt/hddb/dataTFGIvanVerdugo")

def _progress(message):
    """Print a progress message immediately, even when stdout is buffered."""
    print(f"[progress] {message}", flush=True)

def _parser():
    """ Build the parser for the evaluation workflow """
    parser = argparse.ArgumentParser(description=__doc__)

    # Identify the dataset and scene that will be evaluated
    parser.add_argument("--dataset", choices=["replica", "scannetpp"], required=True)
    parser.add_argument("--scene", required=True)

    # Define the paths used by the launcher and by the Docker mounts
    parser.add_argument("--data-root", type=Path, default=None, help="dataset path root, something like .../scannetpp")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None, help="Gaussian model directory to reuse, if exists")
    parser.add_argument(
        "--reuse-from", type=Path, default=None,
        help="previous evaluation root from which GT2D masks and voting data are copied",
    )

    # Select the source of the 2D masks and the container images that produce them
    parser.add_argument("--mask-source", choices=["yolo", "gt2d", "both"], default="yolo")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--train-image", default="tfgivanverdugo/semantic-fusion-gs-train:cuda11.6")
    parser.add_argument("--fusion-image", default="tfgivanverdugo/semantic-fusion-fusion:cuda11.6")
    parser.add_argument("--colmap-image", default="tfgivanverdugo/semantic-fusion-colmap:3.13.0-cpu")

    # Configure dataset preparation and Gaussian training
    parser.add_argument("--sequence-name", default="Sequence_2")
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=None,
        help="training image scale: 1 is original, 2 is half width and height")
    parser.add_argument("--train-data-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--vote-data-device", choices=["cuda", "cpu"], default="cpu")

    # Configure mask generation, vote accumulation and threshold selection
    parser.add_argument("--yolo-conf", type=float, default=0.75)
    parser.add_argument("--size-measure", choices=["max", "gmean", "l2"], default="max")
    parser.add_argument("--hysteresis-gamma", type=float, default=0.8)
    parser.add_argument("--hysteresis-radius", type=float, default=0.05)
    parser.add_argument(
        "--background-mode",
        choices=["all_non_target", "explicit_background", "confidence_weighted"],
        default="confidence_weighted",
        help="How 2D non-target evidence is constructed",
    )
    parser.add_argument("--background-confidence", type=float, default=0.25,
        help="Confidence assigned to pixels with semantic label zero")
    parser.add_argument(
        "--background-view-policy", choices=["target_views", "all_views"],
        default="target_views",
        help="Use only views containing target pixels or every matched view",
    )
    parser.add_argument("--betas", nargs="+", type=float, required=True,
        help="Beta values to evaluate for every target class")

    # Configure transfer methods, background competition and predicted-label weighting
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--min-share", type=float, default=0.5)
    parser.add_argument(
        "--mesh-to-gaussian-transfer",
        choices=["radius_vote", "nearest_neighbor_label"],
        default="radius_vote",
    )
    parser.add_argument(
        "--gaussian-to-mesh-transfer",
        choices=["radius_vote", "nearest_neighbor_label"],
        default="radius_vote",
    )
    parser.add_argument("--min-opacity", type=float, default=0.1)
    parser.add_argument(
        "--gaussian-to-mesh-background-competes",
        dest="gaussian_to_mesh_background_competes",
        action="store_true", default=True,
        help="Include background votes in predicted mesh labels",
    )
    parser.add_argument(
        "--no-gaussian-to-mesh-background-competes",
        dest="gaussian_to_mesh_background_competes",
        action="store_false",
        help="Disable background competition in predicted mesh labels",
    )
    parser.add_argument(
        "--mesh-to-gaussian-background-competes",
        dest="mesh_to_gaussian_background_competes", action="store_true",
        default=True,
        help="Use background votes when assigning GT labels to Gaussians",
    )
    parser.add_argument(
        "--no-mesh-to-gaussian-background-competes",
        dest="mesh_to_gaussian_background_competes", action="store_false",
        help="Do not use background votes when assigning GT labels to Gaussians",
    )
    parser.add_argument("--no-opacity-weighting", action="store_true")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--size-penalty", type=float, default=100.0)
    parser.add_argument("--raster-block-size", type=int, default=16)

    # Rebuild cached data instead of reusing files from an earlier run
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_results_to_csv", action="store_true", default=False,
        help="Append validation results and summaries to dataTFGIvanVerdugo/analytics")
    return parser


def _make_scene(args, data_root, support_dir):
    """
    Create a scene instance based on the dataset type and provided arguments

    The returned instance loads the dataset specific mesh, labels and visibility
    information into the common scene representation
    """
    
    if args.dataset == "replica":
        return ReplicaScene(
            data_root, args.scene, args.sequence_name, args.frame_step, seed=3,
            vertex_label_min_share=0.6, visibility_slop=0.05,
        )
    elif args.dataset == "scannetpp":
        return ScannetScene(data_root, args.scene, support_dir)


def _source_names(mask_source):
    """ Determine the list of mask sources """
    if mask_source == "both":
        return ["yolo", "gt2d"]
    return [mask_source]


def _pending_sources(output_root, mask_source, force):
    """Return sources whose final Markdown result still needs to be generated."""
    pending = []
    for source in _source_names(mask_source):
        result_path = output_root / "results" / f"results_{source}.md"
        if result_path.exists() and not force:
            print(f"[skip] {source}: result already exists at {result_path}")
        else:
            pending.append(source)
    return pending


def _mask_classes(mask_dir, classes):
    """Select target class records present in a generated mask directory.

    classes is the collection of TargetClassInfo supported by the scene.
    classes.json can look like this:

    {
    "73": "refrigerator",
    "63": "tv",
    "59": "potted plant",
    "57": "chair",
    "18": "horse",
    "55": "donut",
    "58": "couch",
    "61": "dining table",
    "65": "mouse"
}

    The returned list contains only records whose detector name appears in the
    mask metadata.
    """
    classes_path = mask_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"mask class metadata not found: {classes_path}")

    # Read detector names written by YOLO in classes.json.
    names = set(json.loads(classes_path.read_text()).values())

    # Use the detector name to retrieve the corresponding main
    # class record.
    mapping = target_classes_by_detector(classes)

    # Keep only supported classes
    selected = []
    for name in sorted(names):
        spec = mapping.get(name)
        if spec is not None:
            selected.append(spec)
    return selected


def _classes_with_gt2d_views(mask_dir, classes):
    """Return classes that occur in at least one generated GT2D view."""
    present_ids = set()
    for path in (mask_dir / "semantic").glob("*.png"):
        semantic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if semantic is not None:
            present_ids.update(np.unique(semantic).tolist())
    return [
        spec for spec in classes
        if spec.detector_stored_id in present_ids
    ]


def _prepare_scene(args, scene, runtime, dataset_dir):
    """ Prepare the dataset in the format expected by training and projection """
    # Replica prepares its images and COLMAP text files locally
    if args.dataset == "replica":
        return scene.prepare_dataset(dataset_dir)

            # Scannet++ prepares its COLMAP model through the Docker container.
    elif args.dataset == "scannetpp":
        return scene.prepare_dataset(runtime)


def _generate_gt_masks(args, scene, runtime, output_dir):
    """
    Generate or reuse the dataset specific 2D masks.

    args.force controls whether existing masks are regenerated.
    """

    # Replica can generate its actual GT masks directly from the semantic image sequence
    if args.dataset == "replica":
        runtime.run_fusion_module(
            "evaluation.replica.gt_masks",
            [
                "--data_root", str(scene.data_root),
                "--scene", scene.scene,
                "--sequence_name", scene.sequence.name,
                "--frame_step", str(scene.frame_step),
                "--vertex_label_min_share", str(scene.vertex_label_min_share),
                "--visibility_slop", str(scene.visibility_slop),
                "--output_dir", str(output_dir),
            ] + (["--force"] if args.force else []),
        )

            # Scannet++ renders its masks from the mesh through the fusion container, they will be considered our "GT"
    elif args.dataset == "scannetpp":
        scene.generate_gt_masks(runtime, output_dir, bands=4, force=args.force)


def _generate_yolo_masks(args, runtime, dataset_dir, output_dir):
    """ Generate or reuse YOLO masks for the prepared dataset images """
    # The classes file says whether the mask directory exists
    if (output_dir / "classes.json").exists() and not args.force:
        return

    # Run the detector inside the fusion container.
    runtime.run_fusion(
            "segmentation/generate_mask.py",
        [
            "--images_dir", str(dataset_dir / "images"),
            "--output_root", str(output_dir),
            "--model", str(runtime.repo_root / "yolo26x-seg.pt"),
            "--conf", str(args.yolo_conf),
        ],
    )


def _export_gt_gaussians(args, runtime, model_dir, gt_dir,
                         segmentation_dir, scene, classes):
    """Export the ground-truth-transfer Gaussians into each source class directory."""
    class_specs = [
        f"{safe_name(spec.name_by_detector)}:{scene.class_id(spec.name)}"
        for spec in classes
    ]
    if not class_specs:
        return
    runtime.run_fusion(
        "segmentation/export_gt_gaussians.py",
        [
            "--model_path", str(model_dir),
             "--gt_labels_path", str(gt_dir / "gt_gaussian_labels.npz"),
            "--output_dir", str(segmentation_dir),
            "--loaded_iter", str(args.iterations),
        ] + sum((["--class_spec", item] for item in class_specs), []),
    )


def _run_votes(args, runtime, dataset_dir, model_dir, mask_dir,
               segmentation_dir, classes, reuse_votes=False,
               save_statistics=False):
    """
    Accumulate 2D votes for every target class present in the masks.

    classes is the list TargetClassInfo selected from the mask
    metadata. Existing vote files are reused unless args.force is true.
    """
    for spec in classes:
        # Each selected main class, identified here by its detector name
        # name, receives its own vote directory and cache file.
        safe = safe_name(spec.name_by_detector)
        vote_path = segmentation_dir / safe / f"voting_data_{safe}.pt"
        statistics_path = segmentation_dir / safe / "vote_statistics.json"
        if (vote_path.exists() and (not args.force or reuse_votes) and
                (not save_statistics or statistics_path.exists())):
            continue

        # Accumulate votes for this main class using masks whose pixels contain
        # stored detector-mask IDs.
        command = [
            "--model_path", str(model_dir),
            "--mask_dir", str(mask_dir),
            "--output_dir", str(segmentation_dir),
            "--target_class", spec.name_by_detector,
            "--loaded_iter", str(args.iterations),
            "--raster_block_size", str(args.raster_block_size),
                "--sigma", str(args.sigma),
            "--size_penalty", str(args.size_penalty),
            "--size_measure", str(args.size_measure),
            "--source_path", str(dataset_dir),
            "--data_device", str(args.vote_data_device),
        ]
        if save_statistics:
            command += [
                "--statistics_path",
                str(statistics_path),
            ]
        runtime.run_fusion(
            "segmentation/accumulate_votes.py", command + [
                "--background_mode", str(args.background_mode),
                "--background_confidence", str(args.background_confidence),
                "--background_view_policy", str(args.background_view_policy),
            ],
        )


def _run_thresholds(args, runtime, model_dir, segmentation_dir, classes):
    """
    Create labeled Gaussian files for every class and beta value

    The returned list contains the beta values used
    Existing labeled files are reused unless args.force is true.
    """

    betas = list(args.betas)
    for spec in classes:
        # Thresholding can only start after vote accumulation produced its file.
        safe = safe_name(spec.name_by_detector)
        vote_path = segmentation_dir / safe / f"voting_data_{safe}.pt"
        if not vote_path.exists():
            continue
        for beta in betas:
            output = segmentation_dir / safe / (
                f"labeled_gaussians_{safe}_beta{str(beta).replace('.', '_')}.ply"
            )
            if output.exists() and not args.force:
                continue

            command = [
                "--voting_data_path", str(vote_path),
                "--model_path", str(model_dir),
                "--output_dir", str(segmentation_dir),
                "--target_class", spec.name_by_detector,
                "--beta", str(beta),
                "--loaded_iter", str(args.iterations),
                "--hysteresis_gamma", str(args.hysteresis_gamma),
                "--hysteresis_radius", str(args.hysteresis_radius),
            ]
            runtime.run_fusion("segmentation/threshold_labels.py", command)
    return betas


def _evaluate_scene(args, scene, gaussians_near_a_vertex, gaussian_labels,
                     full_xyz, full_opacity,
                     classes,
                     segmentation_dir, betas, results_dir, source):
    """
    Evaluate one mask source and write its JSON and markdown results

    betas is the beta threshold grid
    """

    per_class = {}
    per_class_by_beta = {}
    scene_started = time.perf_counter()
    _progress(
        f"Evaluation {source}: {len(classes)} classes, "
        f"{len(betas)} beta value(s)"
    )
    for class_index, spec in enumerate(classes, start=1):
        class_started = time.perf_counter()
        _progress(
            f"{source}: class {class_index}/{len(classes)} "
            f"'{spec.name}' - calculating GT transfer"
        )

        # Calculate the GT-transfer reference before evaluating predictions.
        base = metrics.evaluate_class(
            scene, gaussians_near_a_vertex, gaussian_labels, full_xyz,
            full_opacity, spec, None,
            args.tau, args.min_share, not args.no_opacity_weighting,
            args.min_opacity, args.gaussian_to_mesh_background_competes,
            args.gaussian_to_mesh_transfer,
        )

        ground_truth_transfer_metrics = base["ground_truth_transfer_iou"]
        _progress(
            f"{source}: class '{spec.name}' GT transfer ready "
            f"({time.perf_counter() - class_started:.1f}s)"
        )

        sweep = {}
        safe = safe_name(spec.name_by_detector) # Safe detector name for file names.
        for beta_index, beta in enumerate(betas, start=1):
            beta_started = time.perf_counter()
            _progress(
                f"{source}: class '{spec.name}' beta "
                f"{beta_index}/{len(betas)} ({beta}) - evaluating"
            )

            # A missing labeled file represents an empty prediction for this
            # class and beta, so its Ground Truth instances still contribute
            # false negatives to the metrics.
            path = segmentation_dir / safe / (
                f"labeled_gaussians_{safe}_beta{str(beta).replace('.', '_')}.ply"
            )

            if not path.exists():
                predicted_xyz = np.empty((0, 3), dtype=np.float64)
            else:
                predicted_xyz, _ = transfer.load_gaussian_ply(path)

            # Evaluate the predicted Gaussian mesh, including an empty one.
            result = metrics.evaluate_class(
                scene, gaussians_near_a_vertex, gaussian_labels, full_xyz,
                full_opacity, spec,
                predicted_xyz, args.tau, args.min_share,
                not args.no_opacity_weighting, args.min_opacity,
                args.gaussian_to_mesh_background_competes,
                args.gaussian_to_mesh_transfer, ground_truth_transfer_metrics,
            )
            score = result["iou"]["iou"]
            sweep[str(beta)] = {
                "beta": beta,
                "iou": result["iou"],
                "ground_truth_transfer_iou": result["ground_truth_transfer_iou"],
                "relative_iou": (
                    result["iou"]["iou"] /
                    result["ground_truth_transfer_iou"]["iou"]
                    if result["ground_truth_transfer_iou"]["iou"] else 0.0
                ),
                "score": score,
            }
            per_class_by_beta.setdefault(str(beta), {})[spec.name] = result
            _progress(
                f"{source}: class '{spec.name}' beta {beta} ready "
                f"(IoU={score:.4f}, {time.perf_counter() - beta_started:.1f}s)"
            )

        _progress(
            f"{source}: class '{spec.name}' finished "
            f"({time.perf_counter() - class_started:.1f}s)"
        )

        # Store the complete beta sweep for this target class.
        per_class[spec.name] = {
            "name_by_detector": spec.name_by_detector,
            "sweep": sweep,
        }

    # Aggregate independently for every requested beta.
    metrics_by_beta = {
        beta: metrics.aggregate(beta_classes)
        for beta, beta_classes in per_class_by_beta.items()
    }

    # Save the scene name, evaluation masks, parameters and metrics to JSON and markdown
    result = {
        "dataset": scene.dataset,
        "scene": scene.scene,
        "mask_source": source,
        "support": {
            "vertices_evaluated": int(scene.evaluation_mask.sum()),
        },
        "parameters": {
            "size_measure": args.size_measure,
            "hysteresis_gamma": args.hysteresis_gamma,
            "hysteresis_radius": args.hysteresis_radius,
            "background_mode": args.background_mode,
            "background_confidence": args.background_confidence,
            "background_view_policy": args.background_view_policy,
            "betas": betas,
            "sigma": args.sigma,
            "size_penalty": args.size_penalty,
            "tau": args.tau,
            "min_share": args.min_share,
            "mesh_to_gaussian_transfer": args.mesh_to_gaussian_transfer,
            "gaussian_to_mesh_transfer": args.gaussian_to_mesh_transfer,
            "opacity_weighted": not args.no_opacity_weighting,
            "gaussian_to_mesh_background_competes": args.gaussian_to_mesh_background_competes,
            "mesh_to_gaussian_background_competes": args.mesh_to_gaussian_background_competes,
        },
        "metrics_by_beta": metrics_by_beta,
        "per_class": per_class,
    }
    reporting.write_result(results_dir, result)
    _progress(
        f"Evaluation {source} finished in "
        f"{time.perf_counter() - scene_started:.1f}s"
    )
    return result


def main():
    """ Run preparation, mask generation, voting, thresholding and evaluation """
    args = _parser().parse_args()
    if not 0.0 <= args.background_confidence <= 1.0:
        raise ValueError("--background-confidence must be in [0, 1]")
    if any(beta < 0.0 or beta > 1.0 for beta in args.betas):
        raise ValueError("all --betas must be in [0, 1]")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be greater than zero")
    if args.tau <= 0:
        raise ValueError("--tau must be greater than zero")
    if not 0.0 <= args.min_share <= 1.0:
        raise ValueError("--min-share must be in [0, 1]")
    if args.hysteresis_gamma < 0.0:
        raise ValueError("--hysteresis-gamma must be non-negative")
    if args.hysteresis_radius <= 0.0:
        raise ValueError("--hysteresis-radius must be greater than zero")
    if args.sigma < 0.0:
        raise ValueError("--sigma must be non-negative")
    if args.size_penalty <= 0.0:
        raise ValueError("--size-penalty must be greater than zero")
    if args.raster_block_size <= 0:
        raise ValueError("--raster-block-size must be greater than zero")

    # Determine the data root directory and resolve it to an absolute path.
    data_root = (
        args.data_root
        if args.data_root is not None
        else DEFAULT_DATA_ROOT / args.dataset
    ).resolve() # Resolve the data root path to an absolute path

    # Determine the output root directory and resolve it to an absolute path.
    output_root = (
        args.output_root
        if args.output_root is not None
        else data_root / "evaluation" / args.scene
    ).resolve() # Resolve the output root path to an absolute path

    # Check if the output root is within the data root
    try:
        output_root.relative_to(data_root)
    except ValueError:
        raise ValueError("--output-root must be inside --data-root")

    run_id = str(uuid.uuid4())
    analytics_store = (
        AnalyticsStore(data_root.parent / "analytics")
        if args.save_results_to_csv else None
    )

    # Prepare paths for several outputs
    dataset_dir = output_root / "dataset"
    model_dir = output_root / "model"
    masks_gt = output_root / "masks_gt2d"
    masks_yolo = output_root / "masks_yolo"
    segmentation_root = output_root / "segmentation"
    gt_dir = output_root / "gt"
    results_dir = output_root / "results"

    # Resolve training defaults before checking whether an existing report is valid.
    if args.resolution is None:
        args.resolution = 2 if args.dataset == "scannetpp" else 1
    if args.train_data_device is None:
        args.train_data_device = "cpu" if args.dataset == "scannetpp" else "cuda"
    parameters = cache.run_parameters(args, data_root)

    # Do not initialize scenes, containers or caches when the requested report exists.
    pending_sources = _pending_sources(output_root, args.mask_source, args.force)
    if not pending_sources:
        cache.prepare_run_metadata(
            output_root, parameters, False, _source_names(args.mask_source),
        )
        print("[skip] All requested detector results already exist")
        return
    args.mask_source = (
        pending_sources[0] if len(pending_sources) == 1 else "both"
    )
    run_started = time.perf_counter()

    # Initialize the Docker runtime with the provided arguments.
    runtime = Runtime(
        args.repo_root, data_root, args.train_image,
        args.fusion_image, args.colmap_image,
    )

    # Create a scene instance for the selected dataset.
    scene_instance = _make_scene(args, data_root, masks_gt)

    # Prepare metadata and optionally import reusable scene caches.
    cache.prepare_run_metadata(output_root, parameters, args.force, pending_sources)
    reused_gt_masks, reused_vote_sources = cache.copy_reusable_data(
        args, output_root, parameters,
    )
    dataset_dir = _prepare_scene(args, scene_instance, runtime, dataset_dir)
    model_dir = cache.resolve_model_dir(args, data_root, output_root)

    # Generate ground-truth masks.
    if not reused_gt_masks:
        _generate_gt_masks(args, scene_instance, runtime, masks_gt)
    if args.mask_source in {"yolo", "both"}:
        _generate_yolo_masks(args, runtime, dataset_dir, masks_yolo)

    # Load the common scene data and train only when there is no model.
    scene = scene_instance.load_data()
    evaluation_classes = _classes_with_gt2d_views(masks_gt, scene.classes)
    model_ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    if not model_ply.exists():
        runtime.run_train(dataset_dir, model_dir, args.iterations,
                          args.resolution, args.train_data_device)
    if not model_ply.exists():
        raise FileNotFoundError(f"trained Gaussian model missing: {model_ply}")

    # Build or reuse the mesh and Gaussian neighborhoods and GT labels.
    gaussians_near_a_vertex, gaussian_labels = ground_truth.build(
        scene, model_ply, gt_dir, args.tau, args.min_share,
        args.mesh_to_gaussian_background_competes,
        args.mesh_to_gaussian_transfer, args.force,
    )
    full_xyz, full_opacity = transfer.load_gaussian_ply(model_ply)

    if analytics_store is not None:
        scene_id = f"{scene.dataset}:{scene.scene}"
        analytics_store.append("runs", {
            "run_id": run_id,
            "created_at": utc_now(),
            "status": "running",
            "dataset": scene.dataset,
            "scene_id": scene_id,
            "scene_name": scene.scene,
            "split": args.split,
            "source": args.mask_source,
            "output_root": str(output_root),
            "model_root": str(model_dir),
        })
        analytics_store.append("run_parameters", {
            "run_id": run_id,
            **parameters,
            "mask_source": args.mask_source,
        })
        record_scene_analytics(analytics_store, args, scene, scene_id)

    # Evaluate each mask source (yolo or 2dgt) independently.
    scene_results = {}
    for source in _source_names(args.mask_source):
        # Select the mask directory
        mask_dir = masks_yolo if source == "yolo" else masks_gt

        # Select the segmentation directory for this mask source
        source_dir = segmentation_root / source

        if analytics_store is not None:
            analytics_store.append("run_sources", {
                "run_id": run_id,
                "source": source,
                "mask_directory": str(mask_dir),
                "segmentation_directory": str(source_dir),
            })

        # Select the target classes present in the mask directory
        vote_classes = _mask_classes(mask_dir, evaluation_classes)

        # Export the clean GT-transfer Gaussians into the prediction directories.
        _export_gt_gaussians(
            args, runtime, model_dir, gt_dir, source_dir, scene, vote_classes,
        )

        # Accumulate votes
        _run_votes(
            args, runtime, dataset_dir, model_dir, mask_dir, source_dir, vote_classes,
            source in reused_vote_sources,
            analytics_store is not None,
        )

        # Threshold the votes and produce labeled Gaussian files
        betas = _run_thresholds(args, runtime, model_dir, source_dir, vote_classes)

        # Evaluate every requested beta for the selected mask source.
        scene_results[source] = _evaluate_scene(
            args, scene, gaussians_near_a_vertex, gaussian_labels,
            full_xyz, full_opacity, evaluation_classes,
            source_dir, betas, results_dir, source,
        )
        if analytics_store is not None:
            record_source_analytics(
                analytics_store, run_id, source, scene, evaluation_classes, betas,
                source_dir, scene_results[source], model_ply,
            )

    # Update the source summary without dropping results from another detector.
    summary_path = results_dir / "results.json"
    previous_results = {}
    if summary_path.exists():
        previous_results = json.loads(summary_path.read_text())
    previous_results.update(scene_results)
    summary_path.write_text(json.dumps(previous_results, indent=2, default=str))
    if analytics_store is not None:
        analytics_store.append("runs", {
            "run_id": run_id,
            "created_at": utc_now(),
            "status": "completed",
            "dataset": scene.dataset,
            "scene_id": f"{scene.dataset}:{scene.scene}",
            "scene_name": scene.scene,
            "split": args.split,
            "source": args.mask_source,
            "output_root": str(output_root),
            "model_root": str(model_dir),
        })
    _progress(f"Run finished in {time.perf_counter() - run_started:.1f}s")
    print(json.dumps({name: value["metrics_by_beta"]
                      for name, value in scene_results.items()}, indent=2))


if __name__ == "__main__":
    main()
