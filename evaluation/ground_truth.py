"""Common ground-truth caches and mesh/Gaussian label transfers."""

import json

import numpy as np
from scipy.spatial import cKDTree

from .common import ensure_dir
from . import transfer


class GroundTruthCache:
    """Cached neighborhoods and labels used by the evaluation metrics.

    ``mesh_to_gaussian`` contains the radius-neighbor graph used for prediction
    transfer. ``gaussian_labels`` stores the semantic ground-truth label
    assigned to every Gaussian, and ``gaussian_instances`` stores the matching
    mesh instance when one can be identified.
    """

    def __init__(self, mesh_to_gaussian, gaussian_labels, gaussian_instances):
        """Store the prediction neighborhood and Gaussian labels."""
        self.mesh_to_gaussian = mesh_to_gaussian
        self.gaussian_labels = gaussian_labels
        self.gaussian_instances = gaussian_instances


def _metadata(scene, gaussian_ply, tau, min_share, background_competes,
              transfer_method):
    """Build the metadata used to validate the ground-truth cache."""
    # File size and modification time detect a changed Gaussian model quickly.
    stat = gaussian_ply.stat()
    return {
        # Increase this value when the meaning of the GT cache changes.
        "evaluation_scope_version": 1,
        "dataset": scene.dataset,
        "scene": scene.scene,
        "vertices": int(len(scene.vertices)),
        "gaussians": int(len(transfer.load_gaussian_ply(gaussian_ply)[0])),
        "ply_size": int(stat.st_size),
        "ply_mtime_ns": int(stat.st_mtime_ns),
        "tau": float(tau),
        "min_share": float(min_share),
        "background_competes": bool(background_competes),
        "transfer_method": transfer_method,
    }


def _needs_rebuild(meta_path, expected, force):
    """Return whether the cache must be rebuilt or can be reused."""
    # ``force`` takes priority over every existing cache file.
    if force or not meta_path.exists():
        return True
    try:
        # A metadata mismatch means that at least one cached array is stale.
        return json.loads(meta_path.read_text()) != expected
    except (OSError, ValueError):
        return True


def _gaussian_instance_labels(scene, gaussian_labels, gaussian_to_mesh):
    """Assign each labeled Gaussian the strongest matching mesh instance."""
    # The CSR arrays describe which mesh vertices are inside each Gaussian row.
    indptr, indices, _ = gaussian_to_mesh
    output = np.full(len(gaussian_labels), -1, dtype=np.int64)
    for gaussian_index, semantic in enumerate(gaussian_labels):
        # Invalid semantic labels cannot receive a valid instance label.
        if semantic < 0:
            continue
        start, end = indptr[gaussian_index], indptr[gaussian_index + 1]
        neighbors = indices[start:end]
        if len(neighbors) == 0:
            continue

        # Only vertices with the same semantic class and a valid instance vote.
        valid = ((scene.semantic_labels[neighbors] == semantic) &
                 (scene.instance_labels[neighbors] >= 0))
        if not np.any(valid):
            continue

        # Use the most frequent matching instance among the neighboring vertices.
        values, counts = np.unique(scene.instance_labels[neighbors][valid],
                                   return_counts=True)
        output[gaussian_index] = values[int(np.argmax(counts))]
    return output


def build(scene, gaussian_ply, gt_dir, tau, min_share, background_competes,
          transfer_method="symmetric", force=False):
    """Build or reuse the geometry and ground-truth label caches.

    The cache contains both directions of the radius neighborhoods, semantic
    labels for the full Gaussian model, and the corresponding instance
    labels. The transfer method chooses between the symmetric radius vote and
    the legacy nearest-vertex assignment. Background competition controls
    whether non-target mesh labels participate in the GT Gaussian vote.

    ``force`` is a boolean flag. When it is true, the cached files are rebuilt
    even if their metadata matches the current scene and model.
    """
    # Create the cache directory before reading or writing any cache file.
    ensure_dir(gt_dir)
    meta_path = gt_dir / "cache_meta.json"
    expected = _metadata(scene, gaussian_ply, tau, min_share,
                         background_competes, transfer_method)
    rebuild = _needs_rebuild(meta_path, expected, force)

    # Keep each cache component in a separate file so missing pieces can be rebuilt.
    mesh_to_gaussian_path = gt_dir / "mesh_to_gaussian_neighbors.npz"
    gaussian_to_mesh_path = gt_dir / "gaussian_to_mesh_neighbors.npz"
    gaussian_labels_path = gt_dir / "gt_gaussian_labels.npz"
    gaussian_instances_path = gt_dir / "gt_gaussian_instances.npz"

    # Gaussian centers are needed for both neighborhood directions.
    full_xyz, _ = transfer.load_gaussian_ply(gaussian_ply)
    if rebuild or not mesh_to_gaussian_path.exists():
        # Find Gaussians near every mesh vertex for prediction transfer.
        mesh_to_gaussian = transfer.build_radius_neighbors(
            scene.vertices, cKDTree(full_xyz), tau,
        )
        transfer.save_neighbors(mesh_to_gaussian_path, mesh_to_gaussian)
    else:
        # Reuse the saved graph when the model and transfer settings match.
        mesh_to_gaussian = transfer.load_neighbors(mesh_to_gaussian_path)

    if rebuild or not gaussian_to_mesh_path.exists():
        # Find mesh vertices near every Gaussian for GT label transfer.
        gaussian_to_mesh = transfer.build_radius_neighbors(
            full_xyz, cKDTree(scene.vertices), tau,
        )
        transfer.save_neighbors(gaussian_to_mesh_path, gaussian_to_mesh)
    else:
        # Reuse the opposite-direction graph when it is already available.
        gaussian_to_mesh = transfer.load_neighbors(gaussian_to_mesh_path)

    if rebuild or not gaussian_labels_path.exists():
        # Keep target semantic labels and mark every other vertex as background.
        reference_labels = np.where(scene.semantic_labels >= 0,
                                    scene.semantic_labels, -1).astype(np.int64)
        classes = np.arange(len(scene.classes), dtype=np.int64)
        if transfer_method == "legacy":
            # The legacy method assigns the nearest mesh label within the radius.
            distances, nearest = cKDTree(scene.vertices).query(full_xyz, k=1)
            gaussian_labels = np.where(
                (distances <= tau) & (reference_labels[nearest] >= 0),
                reference_labels[nearest], -1,
            ).astype(np.int64)
        else:
            # The current method uses weighted votes from all nearby mesh vertices.
            gaussian_labels = transfer.radius_label_vote(
                len(full_xyz), gaussian_to_mesh, reference_labels,
                np.ones(len(reference_labels), dtype=np.float64), classes,
                min_share, background_competes,
            )
        np.savez_compressed(gaussian_labels_path, labels=gaussian_labels)
    else:
        # Load semantic labels generated by an earlier compatible run.
        gaussian_labels = np.load(gaussian_labels_path)["labels"]

    if rebuild or not gaussian_instances_path.exists():
        # Match each semantic Gaussian to the strongest compatible mesh instance.
        gaussian_instances = _gaussian_instance_labels(
            scene, gaussian_labels, gaussian_to_mesh,
        )
        np.savez_compressed(gaussian_instances_path, instances=gaussian_instances)
    else:
        # Reuse the cached instance labels when the metadata is still valid.
        gaussian_instances = np.load(gaussian_instances_path)["instances"]

    # Do not mark the cache complete until every array has been written.
    meta_path.write_text(json.dumps(expected, indent=2))
    return GroundTruthCache(
        mesh_to_gaussian=mesh_to_gaussian,
        gaussian_labels=gaussian_labels,
        gaussian_instances=gaussian_instances,
    )
