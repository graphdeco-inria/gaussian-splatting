# Run metadata and reusable evaluation caches

import json
import shutil

from .common import ensure_dir


VOTE_CACHE_KEYS = [
    "dataset", "scene", "data_root", "sequence_name", "frame_step",
    "iterations", "resolution", "size_measure", "background_mode",
    "background_confidence", "background_view_policy", "sigma",
    "size_penalty", "raster_block_size", "vote_data_device",
]

GT_MASK_CACHE_KEYS = [
    "dataset", "scene", "data_root", "sequence_name", "frame_step",
]


def _has_model(model_dir, iterations):
    """ Check if the Gaussian model exists for the requested iteration """
    return (model_dir / "point_cloud" / f"iteration_{iterations}" /
            "point_cloud.ply").exists()


def resolve_model_dir(args, data_root, output_root):
    """Determine the Gaussian model directory to use for evaluation"""

    # If a model root is provided, check if it has the requested iteration.
    if args.model_root is not None:
        model_dir = args.model_root.resolve()
        if not _has_model(model_dir, args.iterations):
            raise FileNotFoundError(
                f"Gaussian model missing for iteration {args.iterations}: "
                f"{model_dir}"
            )
        return model_dir

    # If no model root was provided, check the current output directory.
    output_model = output_root / "model"
    if _has_model(output_model, args.iterations):
        return output_model

    # If neither location has the model, check the dataset's standard location.
    if args.dataset == "replica":
        conventional_models = [
            data_root / args.scene / "eval_output" / "gs_model",
        ]
    else:
        conventional_models = [args.repo_root / "output" / args.scene]
    for conventional_model in conventional_models:
        if _has_model(conventional_model, args.iterations):
            print(f"[model] Reusing existing Gaussian model: {conventional_model}")
            return conventional_model

    # If no existing model is found, return the output path for future training.
    return output_model


def run_parameters(args, data_root):
    """
    Prepare the parameters used to validate reusable evaluation results

    The returned dictionary is written to the output root and used to detect
    incompatible cached results.
    """
    return {
        "evaluation_scope_version": 3, # Increase this value when you don't want to reuse cached results from previous runs
        "dataset": args.dataset,
        "scene": args.scene,
        "data_root": str(data_root),
        "sequence_name": args.sequence_name,
        "frame_step": args.frame_step,
        "iterations": args.iterations,
        "resolution": args.resolution,
        "train_data_device": args.train_data_device,
        "yolo_conf": args.yolo_conf,
        "size_measure": args.size_measure,
        "hysteresis_gamma": args.hysteresis_gamma,
        "hysteresis_radius": args.hysteresis_radius,
        "background_mode": args.background_mode,
        "background_confidence": args.background_confidence,
        "background_view_policy": args.background_view_policy,
        "betas": list(args.betas),
        "tau": args.tau,
        "min_share": args.min_share,
        "mesh_to_gaussian_transfer": args.mesh_to_gaussian_transfer,
        "gaussian_to_mesh_transfer": args.gaussian_to_mesh_transfer,
        "min_opacity": args.min_opacity,
        "gaussian_to_mesh_background_competes": args.gaussian_to_mesh_background_competes,
        "mesh_to_gaussian_background_competes": args.mesh_to_gaussian_background_competes,
        "opacity_weighting": not args.no_opacity_weighting,
        "sigma": args.sigma,
        "size_penalty": args.size_penalty,
        "raster_block_size": args.raster_block_size,
        "vote_data_device": args.vote_data_device,
    }


def prepare_run_metadata(output_root, parameters, force, sources):
    """
    Prepare the output directory and write run parameters to JSON files

    force is boolean. When enabled, only artifacts owned by the requested
    mask sources are removed before the new run starts.
    """
    common_metadata_path = output_root / "run_parameters.json"
    common_previous = None
    if common_metadata_path.exists():
        common_previous = json.loads(common_metadata_path.read_text())
    other_sources_have_results = any(
        (output_root / "results" / f"results_{source}.md").exists()
        for source in {"yolo", "gt2d"} - set(sources)
    )

    # Metadata is kept per mask source so YOLO and GT2D can share an output root.
    for source in sources:
        metadata_path = output_root / f"run_parameters_{source}.json"
        previous = None
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
        elif ((output_root / "results" / f"results_{source}.md").exists() and
              (output_root / "run_parameters.json").exists()):
            
            # Compatibility with output directories created before source metadata.
            previous = json.loads((output_root / "run_parameters.json").read_text())
        elif (output_root / "results" / f"results_{source}.md").exists():
            raise RuntimeError(
                f"{output_root} has a result for {source} but no run metadata"
            )

        if previous is not None and previous != parameters and not force:
            raise RuntimeError(
                f"{output_root} was created with different parameters for {source}; "
                "use a new --output-root or pass --force"
            )

        if force:
            # Only remove artifacts owned by the requested detector.
            mask_dir = output_root / ("masks_gt2d" if source == "gt2d" else "masks_yolo")
            shutil.rmtree(mask_dir, ignore_errors=True)
            shutil.rmtree(output_root / "segmentation" / source, ignore_errors=True)
            for suffix in ["json", "md"]:
                (output_root / "results" / f"results_{source}.{suffix}").unlink(
                    missing_ok=True,
                )

    if (force and common_previous is not None and not other_sources_have_results and
            (common_previous.get("iterations") != parameters["iterations"] or
             common_previous.get("resolution") != parameters["resolution"])):
        
        # These caches are shared, so retain them while another detector result exists
        shutil.rmtree(output_root / "model", ignore_errors=True)
        shutil.rmtree(output_root / "dataset", ignore_errors=True)

    # Create the output root directory and record parameters for later runs.
    ensure_dir(output_root)
    (output_root / "run_parameters.json").write_text(
        json.dumps(parameters, indent=2),
    )
    for source in sources:
        (output_root / f"run_parameters_{source}.json").write_text(
            json.dumps(parameters, indent=2),
        )


def validate_reuse_source(source_root, parameters, keys, artifact_name):
    """ Ensure a reference run is compatible with the requested cache """
    # Read the metadata written by the reference evaluation.
    metadata_path = source_root / "run_parameters.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"reuse source has no run metadata: {metadata_path}"
        )
    source_parameters = json.loads(metadata_path.read_text())

    # Report both missing cache fields and fields with changed values.
    missing = [key for key in keys if key not in source_parameters]
    mismatches = [
        (key, source_parameters.get(key), parameters.get(key))
        for key in keys
        if key in source_parameters and source_parameters[key] != parameters[key]
    ]
    if missing:
        # A cache without all required fields cannot be validated safely.
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Cannot reuse {artifact_name} from {source_root}: reference metadata "
            f"is missing {missing_text}"
        )
    if mismatches:
        # Show the exact changes so an incompatible reference is easy to diagnose.
        changes = "\n".join(
            f"  {key}: {source_value!r} -> {current_value!r}"
            for key, source_value, current_value in mismatches
        )
        raise ValueError(
            f"Cannot reuse {artifact_name} from {source_root}. "
            f"Changed parameters:\n{changes}\n"
            "Use a new reference run or keep the cache parameters identical."
        )


def copy_reusable_data(args, output_root, parameters):
    """ Copy GT2D masks and vote caches from a previous evaluation root """
    
    # Without a reference directory there is nothing to copy.
    if args.reuse_from is None:
        return False, set()

    # The reference and destination must be two different evaluation roots.
    source_root = args.reuse_from.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"reuse source does not exist: {source_root}")
    if source_root == output_root:
        raise ValueError("--reuse-from must be different from --output-root")

    # Validate the parameters that control the generated GT2D masks.
    validate_reuse_source(
        source_root, parameters, GT_MASK_CACHE_KEYS, "GT2D masks",
    )

    # Accept the old directory name when reading an earlier evaluation.
    source_masks = source_root / "masks_gt2d"
    if not source_masks.exists():
        source_masks = source_root / "masks_gt_2d"
    if not source_masks.exists():
        raise FileNotFoundError(
            f"GT2D masks not found in reuse source: {source_root}"
        )
    # Replace the shared mask cache before copying the reference files.
    shutil.rmtree(output_root / "masks_gt2d", ignore_errors=True)
    shutil.copytree(
        source_masks, output_root / "masks_gt2d", dirs_exist_ok=True,
    )
    print(f"cache: Reused GT2D masks from {source_masks}")

    # Vote files have a separate contract from the GT2D mask files.
    validate_reuse_source(
        source_root, parameters, VOTE_CACHE_KEYS, "vote caches",
    )
    reused_sources = set()
    sources = ["yolo", "gt2d"] if args.mask_source == "both" else [args.mask_source]

    # Reuse only the vote data for the requested mask sources.
    for source in sources:
        target_segmentation = output_root / "segmentation" / source
        shutil.rmtree(target_segmentation, ignore_errors=True)
        source_segmentation = source_root / "segmentation" / source
        if not source_segmentation.exists():
            continue
        copied = 0

        # Replace this source before copying its vote files and statistics.
        for vote_path in source_segmentation.glob("*/voting_data_*.pt"):
            target_path = target_segmentation / vote_path.parent.name / vote_path.name
            ensure_dir(target_path.parent)
            shutil.copy2(vote_path, target_path)
            statistics_path = vote_path.parent / "vote_statistics.json"
            if statistics_path.exists():
                shutil.copy2(
                    statistics_path,
                    target_segmentation / vote_path.parent.name / statistics_path.name,
                )
            copied += 1
        print(
            f"cache: Reused {copied} vote files for {source} "
            f"from {source_segmentation}"
        )
        if copied:
            reused_sources.add(source)
    return True, reused_sources
