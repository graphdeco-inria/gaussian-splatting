# The workflow that evaluates both Scannet++ and Replica datasets

import argparse
import json
import shutil
from pathlib import Path

from . import ground_truth, metrics, transfer
from .common import ensure_dir, safe_name, target_classes_by_detector
from .runtime import Runtime

from .replica.scene import ReplicaScene
from .scannetpp.scene import ScanNetScene


DEFAULT_DATA_ROOT = Path("/mnt/hddb/dataTFGIvanVerdugo")
DEFAULT_BETAS = [
    0.01, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15,
    0.2, 0.3, 0.35, 0.5, 1.2,
]


def _parser():
    """ Build the parser for the evaluation workflow """
    parser = argparse.ArgumentParser(description=__doc__)

    # Identify the dataset and scene that will be evaluated
    parser.add_argument("--dataset", choices=["replica", "scannetpp"], required=True)
    parser.add_argument("--scene", required=True)

    # Define the paths used by the launcher and by the Docker mounts
    parser.add_argument("--data-root", type=Path, default=None,
                        help="dataset path root, something like .../scannetpp")
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None,
                        help="Gaussian model directory to reuse, if exists")

    # Select the source of the 2D masks and the container images that produce them
    parser.add_argument("--mask-source", choices=["yolo", "gt2d", "both"],
                        default="yolo")
    parser.add_argument("--train-image",
                        default="tfgivanverdugo/semantic-fusion-gs-train:cuda11.6")
    parser.add_argument("--fusion-image",
                        default="tfgivanverdugo/semantic-fusion-fusion:cuda11.6")
    parser.add_argument("--colmap-image",
                        default="tfgivanverdugo/semantic-fusion-colmap:3.13.0-cpu")

    # Configure dataset preparation and Gaussian training
    parser.add_argument("--sequence-name", default="Sequence_2")
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="training image scale: 1 is original, 2 is half width and height",
    )
    parser.add_argument("--train-data-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--vote-data-device", choices=["cuda", "cpu"], default="cpu")

    # Configure mask generation, vote accumulation and threshold selection
    parser.add_argument("--yolo-conf", type=float, default=0.75)
    parser.add_argument("--size-measure", choices=["max", "gmean", "l2"], default="l2")
    parser.add_argument("--thresh-mode", choices=["class_views", "cameras"],
                        default="class_views")
    parser.add_argument("--hysteresis-gamma", type=float, default=0.5)
    parser.add_argument("--hysteresis-radius", type=float, default=0.05)
    parser.add_argument("--betas", nargs="*", type=float, default=DEFAULT_BETAS)

    # Configure ground-truth transfer and predicted label weighting
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--min-share", type=float, default=0.5)
    parser.add_argument("--gt-transfer", choices=["symmetric", "legacy"],
                        default="symmetric")
    parser.add_argument("--min-opacity", type=float, default=0.1)
    parser.add_argument("--background-competes", dest="background_competes",
                        action="store_true", default=False,
                        help="Include background votes in predicted mesh labels")
    parser.add_argument("--no-background-competes", dest="background_competes",
                        action="store_false",
                        help="Disable background competition in predicted mesh labels")
    parser.add_argument("--gt-background-competes",
                        dest="gt_background_competes", action="store_true",
                        default=True,
                        help="Use background votes when assigning GT labels to Gaussianas")
    parser.add_argument("--no-gt-background-competes",
                        dest="gt_background_competes", action="store_false",
                        help="Do not use background votes when assigning GT labels to Gaussianas")
    parser.add_argument("--no-opacity-weighting", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--size-penalty", type=float, default=100.0)
    parser.add_argument("--raster-block-size", type=int, default=16)
    parser.add_argument("--pq-radius", type=float, default=0.15)
    parser.add_argument("--pq-min-component", type=int, default=20)
    parser.add_argument("--pq-match-iou", type=float, default=0.5)

    # Rebuild cached data instead of reusing files from an earlier run
    parser.add_argument("--force", action="store_true")
    return parser


def _has_model(model_dir, iterations):
    """ Check if the Gaussian model exists for the given number of iterations """
    return (model_dir / "point_cloud" / f"iteration_{iterations}" /
            "point_cloud.ply").exists()


def _resolve_model_dir(args, data_root, output_root):
    """ Determine the Gaussian model directory to use for evaluation """

    # If a model root is provided, check if it has the requested iteration.
    if args.model_root is not None:
        model_dir = args.model_root.resolve()
        if not _has_model(model_dir, args.iterations):
            raise FileNotFoundError(
                f"Gaussian model missing for iteration {args.iterations}: "
                f"{model_dir}"
            )
        return model_dir

    # If no model root was provided, check the current output directory
    output_model = output_root / "model"
    if _has_model(output_model, args.iterations):
        return output_model

    # If neither location has the model, check the dataset's standard location
    if args.dataset == "replica":
        conventional_models = [
            data_root / args.scene / "eval_output" / "gs_model",
        ]
    elif args.dataset == "scannetpp":
        conventional_models = [args.repo_root / "output" / args.scene]
    for conventional_model in conventional_models:
        if _has_model(conventional_model, args.iterations):
            print(f"[model] Reusing existing Gaussian model: {conventional_model}")
            return conventional_model

    # If no existing model is found, return the output path for future training
    return output_model


def _run_parameters(args, data_root):
    """
    Prepare a dictionary of parameters for the current evaluation run

    The returned dictionary is written to the output root and used to detect
    incompatible cached results.
    """
    return {
        "evaluation_scope_version": 2, # Increase this value when you don't want to reuse cached results from previous runs
        "dataset": args.dataset,
        "scene": args.scene,
        "data_root": str(data_root),
        "frame_step": args.frame_step,
        "iterations": args.iterations,
        "resolution": args.resolution,
        "train_data_device": args.train_data_device,
        "yolo_conf": args.yolo_conf,
        "size_measure": args.size_measure,
        "thresh_mode": args.thresh_mode,
        "hysteresis_gamma": args.hysteresis_gamma,
        "hysteresis_radius": args.hysteresis_radius,
        "betas": list(args.betas),
        "tau": args.tau,
        "min_share": args.min_share,
        "gt_transfer": args.gt_transfer,
        "min_opacity": args.min_opacity,
        "background_competes": args.background_competes,
        "gt_background_competes": args.gt_background_competes,
        "opacity_weighting": not args.no_opacity_weighting,
    }


def _prepare_run_metadata(output_root, parameters, force):
    """
    Prepare the output directory and write run parameters to a JSON file

    force is boolean. When enabled, cached masks, segmentation,
    ground truth data and results are removed before the new run starts
    """
    metadata_path = output_root / "run_parameters.json"

    # If metadata exists, check whether the cached files belong to this run
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous != parameters and not force:
            raise RuntimeError(
                f"{output_root} was created with different parameters; "
                "use a new --output-root or pass --force"
            )

        # If force is enabled, remove outputs that depend on the run parameters
        if force:
            for name in ["masks_yolo", "masks_gt2d", "segmentation", "gt", "results"]:
                shutil.rmtree(output_root / name, ignore_errors=True)
            if (previous.get("iterations") != parameters["iterations"] or
                    previous.get("resolution") != parameters["resolution"]):
                # A training change invalidates the model and prepared images
                shutil.rmtree(output_root / "model", ignore_errors=True)
                shutil.rmtree(output_root / "dataset", ignore_errors=True)

    # Create the output root directory and record the parameters for later runs
    ensure_dir(output_root)
    metadata_path.write_text(json.dumps(parameters, indent=2))


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
        return ScanNetScene(data_root, args.scene, support_dir)


def _source_names(mask_source):
    """ Determine the list of mask sources """
    if mask_source == "both":
        return ["yolo", "gt2d"]
    return [mask_source]


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

    # Read the names of the classes that YOLO produced
    names = set(json.loads(classes_path.read_text()).values())

    # Assigns the name to the whole data of the target class
    mapping = target_classes_by_detector(classes)

    # Keep only supported classes
    selected = []
    for name in sorted(names):
        spec = mapping.get(name)
        if spec is not None:
            selected.append(spec)
    return selected


def _prepare_scene(args, scene, runtime, dataset_dir):
    """ Prepare the dataset in the format expected by training and projection """
    # Replica prepares its images and COLMAP text files locally
    if args.dataset == "replica":
        return scene.prepare_dataset(dataset_dir)

    # ScanNet++ prepares its COLMAP model through the Docker container.
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
                "--output_dir", str(output_dir),
            ] + (["--force"] if args.force else []),
        )

    # ScanNet++ renders its masks from the mesh through the fusion container, they will be considered our "GT"
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


def _run_votes(args, runtime, dataset_dir, model_dir, mask_dir,
               segmentation_dir, classes):
    """
    Accumulate 2D votes for every target class present in the masks.

    classes is the list TargetClassInfo selected from the mask
    metadata. Existing vote files are reused unless args.force is true.
    """
    for spec in classes:
        # Each detector class receives its own vote directory and cache file
        safe = safe_name(spec.name_by_detector)
        vote_path = segmentation_dir / safe / f"voting_data_{safe}.pt"
        if vote_path.exists() and not args.force:
            continue

        # Accumulate mask votes for this class inside the fusion container
        runtime.run_fusion(
            "segmentation/accumulate_votes.py",
            [
                "--model_path", str(model_dir),
                "--mask_dir", str(mask_dir),
                "--output_dir", str(segmentation_dir),
                "--target_class", spec.name_by_detector,
                "--loaded_iter", str(args.iterations),
                "--raster_block_size", str(args.raster_block_size),
                "--alpha", str(args.alpha),
                "--size_penalty", str(args.size_penalty),
                "--size_measure", str(args.size_measure),
                "--source_path", str(dataset_dir),
                "--data_device", str(args.vote_data_device),
            ],
        )


def _run_thresholds(args, runtime, model_dir, segmentation_dir, classes):
    """
    Create labeled Gaussian files for every class and beta value

    The returned list contains the beta values used
    Existing labeled files are reused unless args.force is true.
    """

    betas = sorted(set(args.betas))
    for spec in classes:

        # Thresholding can only start after vote accumulation produced its file
        safe = safe_name(spec.name_by_detector)
        vote_path = segmentation_dir / safe / f"voting_data_{safe}.pt"
        if not vote_path.exists():
            continue

        for beta in betas:
            # Every beta produces a separate labeled Gaussian PLY file
            output = segmentation_dir / safe / (
                f"labeled_gaussians_{safe}_beta{str(beta).replace('.', '_')}.ply"
            )
            if output.exists() and not args.force:
                continue

            runtime.run_fusion(
                "segmentation/threshold_labels.py",
                [
                    "--voting_data_path", str(vote_path),
                    "--model_path", str(model_dir),
                    "--output_dir", str(segmentation_dir),
                    "--target_class", spec.name_by_detector,
                    "--beta", str(beta),
                    "--loaded_iter", str(args.iterations),
                    "--thresh_mode", str(args.thresh_mode),
                    "--hysteresis_gamma", str(args.hysteresis_gamma),
                    "--hysteresis_radius", str(args.hysteresis_radius),
                ],
            )
    return betas


def _evaluate_scene(args, scene, cache, full_xyz, full_opacity,
                     segmentation_dir, betas, results_dir, source):
    """
    Evaluate one mask source and write its JSON and markdown results

    betas is the beta threshold grid
    """

    per_class = {}
    for spec in scene.classes:

        # Calculate the representation ceiling before evaluating predictions
        base = metrics.evaluate_class(
            scene, cache, full_xyz, full_opacity, spec, None, None,
            args.tau, args.min_share, not args.no_opacity_weighting,
            args.min_opacity, args.background_competes, 5,
            args.pq_radius, args.pq_min_component, args.pq_match_iou,
        )

        ceiling = {
            "iou": base["ceiling_iou"],
            "pq": base["ceiling_pq"],
        }

        best = None
        sweep = {}
        safe = safe_name(spec.name_by_detector) # Sustitute spaces with underscores for file names
        for beta in betas:

            # A missing labeled file means this class/beta has no prediction
            path = segmentation_dir / safe / (
                f"labeled_gaussians_{safe}_beta{str(beta).replace('.', '_')}.ply"
            )

            if not path.exists():
                continue

            # Load the predicted Gaussian mesh and evaluate it against the ground truth
            predicted_xyz, predicted_opacity = transfer.load_gaussian_ply(path)
            result = metrics.evaluate_class(
                scene, cache, full_xyz, full_opacity, spec,
                predicted_xyz, predicted_opacity, args.tau, args.min_share,
                not args.no_opacity_weighting, args.min_opacity,
                args.background_competes, 5, args.pq_radius,
                args.pq_min_component, args.pq_match_iou, ceiling,
            )

            # Select the beta with the highest IoU for this class and store all results
            score = result["iou"]["iou"]
            sweep[str(beta)] = {
                "iou": result["iou"],
                "pq": result["pq"],
                "score": score,
            }
            if best is None or score > best["score"]:
                best = {"beta": beta, "score": score, "result": result}

        # Preserve the ceiling-only result when no prediction was generated.
        if best is None:
            result = base
            best_beta = None
        else:
            # Select the beta with the highest IoU for this class.
            result = best["result"]
            best_beta = best["beta"]

        # Store the best beta and its evaluation metrics for this target class
        per_class[spec.name] = {
            "name_by_detector": spec.name_by_detector,
            "best_beta": best_beta,
            "gt_instances": result["gt_instances"],
            "iou": result["iou"],
            "ceiling_iou": result["ceiling_iou"],
            "pq": result["pq"],
            "ceiling_pq": result["ceiling_pq"],
            "sweep": sweep,
        }

    # Aggregate after every target class has its selected result
    aggregate = metrics.aggregate(per_class)

    # Calculate the visible and annotated vertex set for the final report
    visible_annotated = (
        scene.annotated & (scene.instance_labels >= 0) & scene.visible
    )

    # Save the scene name, evaluation masks, parameters and metrics to JSON and markdown
    result = {
        "dataset": scene.dataset,
        "scene": scene.scene,
        "mask_source": source,
        "support": {
            "instances_total": int(len(set(scene.instance_labels[scene.instance_labels >= 0]))),
            "instances_visible_annotated": int(
                scene.visible_instance_count()
            ),
            "instances_seen_by_2D_masks": int(len(scene.instances_seen_by_2D_masks)),
            "vertices_evaluated": int(scene.evaluation_mask.sum()),
        },
        "parameters": {
            "size_measure": args.size_measure,
            "thresh_mode": args.thresh_mode,
            "hysteresis_gamma": args.hysteresis_gamma,
            "hysteresis_radius": args.hysteresis_radius,
            "betas": betas,
            "tau": args.tau,
            "min_share": args.min_share,
            "gt_transfer": args.gt_transfer,
            "opacity_weighted": not args.no_opacity_weighting,
            "background_competes": args.background_competes,
            "gt_background_competes": args.gt_background_competes,
        },
        "metrics": aggregate,
        "per_class": per_class,
    }
    _write_result(results_dir, result)
    return result


def _format_metric(value):
    """ Format a metric value for the markdown report """
    return "-" if value is None else f"{value:.4f}"


def _write_result(results_dir, result):
    """ Write one source result as JSON and a compact Markdown report """
    ensure_dir(results_dir)
    tag = result["mask_source"] # Either "yolo" or "gt2d"
    json_path = results_dir / f"results_{tag}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))

    # Format the headline metrics as markdown rows
    metrics_block = result["metrics"]
    lines = [
        f"# {result['dataset']} {result['scene']} ({tag})",
        "",
        "## Metrics",
        "",
    ]

    for name in ["mIoU", "ceiling_mIoU", "relative_mIoU", "global_iou", "mPQ", "mSQ", "mRQ", "ceiling_mPQ"]:
        lines.append(f"{name}: {_format_metric(metrics_block[name])}")
    lines += ["", "## Per class", ""]

    # Add one row for every class included in the source result
    for name, item in result["per_class"].items():
        ceiling = item["ceiling_iou"]["iou"]
        relative = item["iou"]["iou"] / ceiling if ceiling else 0.0
        lines.append(
            f"{name}: beta={item['best_beta']}, "
            f"IoU={item['iou']['iou']:.4f}, "
            f"ceiling IoU={ceiling:.4f}, "
            f"relative IoU={relative:.4f}, "
            f"PQ={_format_metric(item['pq']['pq'])}, "
            f"SQ={_format_metric(item['pq']['sq'])}, "
            f"RQ={_format_metric(item['pq']['rq'])}"
        )

    # Write the markdown report to a file named after the mask source
    (results_dir / f"results_{tag}.md").write_text("\n".join(lines) + "\n")


def main():
    """Run preparation, mask generation, voting, thresholding and evaluation."""
    args = _parser().parse_args()

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

    # Prepare paths for several outputs
    dataset_dir = output_root / "dataset"
    model_dir = output_root / "model"
    masks_gt = output_root / "masks_gt2d"
    masks_yolo = output_root / "masks_yolo"
    segmentation_root = output_root / "segmentation"
    gt_dir = output_root / "gt"
    results_dir = output_root / "results"

    # Initialize the Docker runtime with the provided arguments.
    runtime = Runtime(
        args.repo_root, data_root, args.train_image,
        args.fusion_image, args.colmap_image,
    )

    # Create a scene instance and resolve dataset specific training defaults.
    scene_instance = _make_scene(args, data_root, masks_gt)
    if args.resolution is None:
        args.resolution = 2 if args.dataset == "scannetpp" else 1 # Scannet uses larger DSLR images and therefore defaults to half scale
    if args.train_data_device is None:
        # Keep Replica images on CUDA but avoid large ScanNet++ GPU allocations (kept from when the server had memory shortage)
        args.train_data_device = "cpu" if args.dataset == "scannetpp" else "cuda"

    # Prepare metadata, the scene dataset, and the Gaussian model directory.
    _prepare_run_metadata(output_root, _run_parameters(args, data_root), args.force)
    dataset_dir = _prepare_scene(args, scene_instance, runtime, dataset_dir)
    model_dir = _resolve_model_dir(args, data_root, output_root)

    # Generate ground-truth masks.
    _generate_gt_masks(args, scene_instance, runtime, masks_gt)
    if args.mask_source in {"yolo", "both"}:
        _generate_yolo_masks(args, runtime, dataset_dir, masks_yolo)

    # Load the common scene data and train only when there is no model.
    scene = scene_instance.load_data()
    model_ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    if not model_ply.exists():
        runtime.run_train(dataset_dir, model_dir, args.iterations,
                          args.resolution, args.train_data_device)
    if not model_ply.exists():
        raise FileNotFoundError(f"trained Gaussian model missing: {model_ply}")

    # Build or reuse the mesh and Gaussian neighborhoods and GT labels.
    cache = ground_truth.build(
        scene, model_ply, gt_dir, args.tau, args.min_share,
        args.gt_background_competes, args.gt_transfer, args.force,
    )
    full_xyz, full_opacity = transfer.load_gaussian_ply(model_ply)

    # Evaluate each mask source (yolo or 2dgt) independently.
    scene_results = {}
    for source in _source_names(args.mask_source):

        # Select the mask directory
        mask_dir = masks_yolo if source == "yolo" else masks_gt

        # Select the segmentation directory for this mask source
        source_dir = segmentation_root / source

        # Select the target classes present in the mask directory
        classes = _mask_classes(mask_dir, scene.classes)

        # Accumulate votes
        _run_votes(args, runtime, dataset_dir, model_dir, mask_dir,
                   source_dir, classes)

        # Threshold the votes and produce labeled Gaussian files
        betas = _run_thresholds(args, runtime, model_dir, source_dir, classes)

        # Evaluate the scene with the selected mask source and write its results
        scene_results[source] = _evaluate_scene(
            args, scene, cache, full_xyz, full_opacity, source_dir, betas, results_dir, source,
        )

    # Write the source summary
    (results_dir / "results.json").write_text(
        json.dumps(scene_results, indent=2, default=str),
    )
    print(json.dumps({name: value["metrics"]
                      for name, value in scene_results.items()}, indent=2))


if __name__ == "__main__":
    main()
