"""Shared radius-vote transfers between mesh vertices and Gaussians."""

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-10


def sigmoid(values):
    """Convert opacity logits into values between zero and one."""
    # Gaussian PLY files store opacity before the sigmoid conversion.
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def load_gaussian_ply(path):
    """Load Gaussian centers and opacity values converted to alpha.

    The return value contains one array of 3D centers and one array of opacity
    values in the same Gaussian order.
    """
    from plyfile import PlyData

    # Keep the original Gaussian order because labels are mapped by this order.
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
    names = vertex.data.dtype.names or ()
    if "opacity" in names:
        # Convert stored opacity logits into alpha values used as vote weights.
        opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))
    else:
        # Older PLY files without opacity give every Gaussian equal weight.
        opacity = np.ones(len(xyz), dtype=np.float64)
    return xyz, opacity


def map_subset_indices(full_xyz, subset_xyz):
    """Map subset Gaussian centers back to their positions in the full model."""
    # Labeled PLY files contain a subset of the full model in a new order.
    distance, indices = cKDTree(full_xyz).query(subset_xyz, k=1)
    if len(distance) and float(distance.max()) > 1e-6:
        raise ValueError("labeled Gaussian PLY does not match the full model")
    return indices


def build_radius_neighbors(query_points, reference_tree, radius,
                           chunk_size=75000):
    """Build radius neighborhoods for every query point.

    The result contains row offsets, neighbor indices and distances. Rows
    correspond to query points and the indices refer to points in the search
    tree.
    """
    # Store each chunk separately so large scenes do not require one huge query.
    index_chunks = []
    distance_chunks = []
    count_chunks = []
    for start in range(0, len(query_points), chunk_size):
        end = min(start + chunk_size, len(query_points))
        # Query all points in this chunk against the reference KD-tree.
        lists = reference_tree.query_ball_point(
            query_points[start:end], r=radius, workers=-1,
        )
        # CSR row counts allow the variable-length neighbor lists to be flattened.
        counts = np.fromiter((len(item) for item in lists), dtype=np.int64,
                             count=len(lists))
        count_chunks.append(counts)
        chunk_indices = np.empty(int(counts.sum()), dtype=np.int32)
        chunk_distances = np.empty(int(counts.sum()), dtype=np.float32)
        position = 0
        for query_index, neighbors in enumerate(lists):
            count = len(neighbors)
            if count:
                # Save neighbor indices and their exact distances in flat arrays.
                neighbor_indices = np.asarray(neighbors, dtype=np.int32)
                chunk_indices[position:position + count] = neighbor_indices
                chunk_distances[position:position + count] = np.linalg.norm(
                    reference_tree.data[neighbor_indices] - query_points[start + query_index],
                    axis=1,
                )
            position += count
        index_chunks.append(chunk_indices)
        distance_chunks.append(chunk_distances)

    # Join all chunks and build the CSR row-offset array.
    counts = np.concatenate(count_chunks) if count_chunks else np.empty(0, dtype=np.int64)
    indptr = np.zeros(len(query_points) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.concatenate(index_chunks) if index_chunks else np.empty(0, dtype=np.int32)
    distances = (np.concatenate(distance_chunks)
                 if distance_chunks else np.empty(0, dtype=np.float32))
    return indptr, indices, distances


def save_neighbors(path, csr):
    """Save a neighborhood structure in a compressed NumPy file."""
    # Store the three CSR components under stable names for later reuse.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indptr=csr[0], indices=csr[1], dist=csr[2])


def load_neighbors(path):
    """Load row offsets, neighbor indices and distances from disk."""
    # Return the same tuple shape produced by ``build_radius_neighbors``.
    data = np.load(path)
    return data["indptr"], data["indices"], data["dist"]


def radius_label_vote(n_query, csr, reference_labels, reference_weights,
                      classes, min_share, background_competes):
    """Assign labels with a weighted radius vote and optional abstention.

    ``csr`` contains one neighborhood row per query point. Reference labels
    and weights use the order of the referenced points. ``background_competes``
    is a boolean flag that decides whether non-target labels contribute to the
    vote denominator.
    """
    # Expand the CSR structure into one entry per query/reference edge.
    indptr, indices, distances = csr
    if len(indices) == 0:
        return np.full(n_query, -1, dtype=np.int64)

    edge_query = np.repeat(np.arange(n_query, dtype=np.int64),
                           np.diff(indptr))
    edge_labels = reference_labels[indices]
    edge_weights = (reference_weights[indices].astype(np.float64) /
                    (distances.astype(np.float64) ** 2 + EPS))

    # The denominator either includes or excludes invalid/background labels.
    if background_competes:
        total = np.bincount(edge_query, weights=edge_weights,
                            minlength=n_query)
    else:
        total = np.bincount(
            edge_query,
            weights=np.where(edge_labels >= 0, edge_weights, 0.0),
            minlength=n_query,
        )

    # Start with no class selected; background can be the initial competitor.
    best_score = np.zeros(n_query, dtype=np.float64)
    best_label = np.full(n_query, -1, dtype=np.int64)
    if background_competes:
        best_score = np.bincount(
            edge_query,
            weights=np.where(edge_labels < 0, edge_weights, 0.0),
            minlength=n_query,
        )

    # Compare the weighted score of every target class for every query point.
    for label in classes:
        score = np.bincount(
            edge_query,
            weights=np.where(edge_labels == label, edge_weights, 0.0),
            minlength=n_query,
        )
        better = score > best_score
        best_score[better] = score[better]
        best_label[better] = label

    # Abstain when the winning class does not reach the required vote share.
    share = np.divide(best_score, total, out=np.zeros_like(best_score),
                      where=total > 0)
    accepted = ((total > 0) & (best_score > 0) & (share >= min_share))
    output = np.full(n_query, -1, dtype=np.int64)
    output[accepted] = best_label[accepted]
    return output


def predict_vertex_labels(mesh_xyz, mesh_gaussian_csr, gaussian_labels,
                          gaussian_opacity, tau, min_share,
                          opacity_weighted, min_opacity,
                          background_competes):
    """Transfer Gaussian labels to mesh vertices with a radius vote.

    ``opacity_weighted`` and ``background_competes`` are boolean flags. The
    first decides whether Gaussian opacity affects the vote weights; the
    second decides whether background neighbors compete with target labels.
    Vertices without an accepted label receive the invalid-label value.
    """
    # Select opacity-aware or uniform weights before running the common vote.
    if opacity_weighted:
        weights = np.clip(gaussian_opacity, min_opacity, 1.0)
    else:
        weights = np.ones(len(gaussian_labels), dtype=np.float64)
    # Only valid labels can participate as target classes.
    classes = np.unique(gaussian_labels[gaussian_labels >= 0])
    return radius_label_vote(
        len(mesh_xyz), mesh_gaussian_csr, gaussian_labels, weights, classes,
        min_share, background_competes,
    )
