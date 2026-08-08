# Ground truth caches and mesh/Gaussian label transfers

import json

import numpy as np
from scipy.spatial import cKDTree

from .common import ensure_dir
from . import transfer


def _metadata(scene, gaussian_ply, tau, min_share,
              mesh_to_gaussian_background_competes, mesh_to_gaussian_transfer):
    """ Build the metadata used to validate the GT cache """

    # File size and modification time detect a changed Gaussian model
    stat = gaussian_ply.stat()
    return {
        "evaluation_scope_version": 2, # Increase this value when the meaning of the GT cache changes
        "dataset": scene.dataset,
        "scene": scene.scene,
        "vertices": int(len(scene.vertices)),
        "classes": [item.name for item in scene.classes],
        "gaussians": int(len(transfer.load_gaussian_ply(gaussian_ply)[0])),
        "ply_size": int(stat.st_size),
        "ply_mtime_ns": int(stat.st_mtime_ns),
        "tau": float(tau),
        "min_share": float(min_share),
        "mesh_to_gaussian_background_competes": bool(mesh_to_gaussian_background_competes),
        "mesh_to_gaussian_transfer": mesh_to_gaussian_transfer,
    }


def _needs_rebuild(meta_path, expected, force):
    """ Return whether the cache must be rebuilt or can be reused """

    # force takes priority
    if force or not meta_path.exists():
        return True
    try:
        # A metadata mismatch means that at least one cached array is stale
        return json.loads(meta_path.read_text()) != expected
    except (OSError, ValueError):
        return True


def build(scene, gaussian_ply, gt_dir, tau, min_share, mesh_to_gaussian_background_competes,
          mesh_to_gaussian_transfer="radius_vote", force=False):
    """
    Build or reuse the neighborhoods and the Gaussians GT local semantic labels used for evaluation

    - The cache contains both directions of the radius neighborhoods and semantic labels for the Gaussian model
    - The transfer method chooses between the radius vote and nearest neighbor label assignment

    mesh_to_gaussian_background_competes controls whether non-target mesh labels
    participate when transferring GT labels from the mesh to Gaussians.
    force makes that the cached files are rebuilt even if their metadata matches the current scene and model.
    """

    # Create the cache directory before reading or writing any cache file
    ensure_dir(gt_dir)
    meta_path = gt_dir / "cache_meta.json"
    expected = _metadata(scene, gaussian_ply, tau, min_share, mesh_to_gaussian_background_competes, mesh_to_gaussian_transfer)
    rebuild = _needs_rebuild(meta_path, expected, force)

    # Keep each cache component in a separate file so missing pieces can be rebuilt
    gaussians_near_a_vertex_path = gt_dir / "gaussians_near_a_vertex_neighbors.npz"
    vertices_near_a_gaussian_path = gt_dir / "vertices_near_a_gaussian_neighbors.npz"
    gaussian_labels_path = gt_dir / "gt_gaussian_labels.npz"

    # Gaussian centers are needed for both neighborhood directions
    full_xyz, _ = transfer.load_gaussian_ply(gaussian_ply)

    if rebuild or not gaussians_near_a_vertex_path.exists():

        # Find Gaussians near every mesh vertex for the Gaussian-to-mesh transfer.
        gaussians_near_a_vertex = transfer.build_radius_neighbors(scene.vertices, cKDTree(full_xyz), tau)
        transfer.save_neighbors(gaussians_near_a_vertex_path, gaussians_near_a_vertex)

    else:
        # Reuse the saved graph when the model and transfer settings match
        gaussians_near_a_vertex = transfer.load_neighbors(gaussians_near_a_vertex_path)

    if rebuild or not vertices_near_a_gaussian_path.exists():

        # Find vertices near every Gaussian for the mesh-to-Gaussian transfer.
        vertices_near_a_gaussian = transfer.build_radius_neighbors(full_xyz, cKDTree(scene.vertices), tau)
        transfer.save_neighbors(vertices_near_a_gaussian_path, vertices_near_a_gaussian)

    else:
        vertices_near_a_gaussian = transfer.load_neighbors(vertices_near_a_gaussian_path)

    if rebuild or not gaussian_labels_path.exists():

        # Keep target local semantic IDs and mark every other vertex as background, -1 in the local ID space
        reference_labels = np.where(scene.semantic_labels >= 0, scene.semantic_labels, -1).astype(np.int64)
        classes = np.arange(len(scene.classes), dtype=np.int64)

        if mesh_to_gaussian_transfer == "nearest_neighbor_label":

            # The nearest-neighbor method assigns to the gaussian the nearest vertex local label within the radius
            distances, nearest = cKDTree(scene.vertices).query(full_xyz, k=1) # Find the nearest mesh vertex for every Gaussian center
            gaussian_labels = np.where((distances <= tau) & (reference_labels[nearest] >= 0), reference_labels[nearest], -1,).astype(np.int64)

        else:
            # The radius-vote method uses weighted votes from all nearby mesh vertices
            gaussian_labels = transfer.radius_label_vote(
                len(full_xyz), vertices_near_a_gaussian, reference_labels,
                np.ones(len(reference_labels), dtype=np.float64), classes,
                min_share, mesh_to_gaussian_background_competes,
            )
        np.savez_compressed(gaussian_labels_path, labels=gaussian_labels)

    else:
        # Load local semantic labels generated by an earlier compatible run
        gaussian_labels = np.load(gaussian_labels_path)["labels"]

    # Record the metadata after all required cache components are ready
    meta_path.write_text(json.dumps(expected, indent=2))
    return gaussians_near_a_vertex, gaussian_labels
