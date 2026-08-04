""" Build reusable radius neighborhoods between a mesh and Gaussian model """

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-10


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def load_gaussian_ply(path):
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    names = vertex.data.dtype.names or ()
    opacity = sigmoid(vertex["opacity"]) if "opacity" in names else np.ones(len(xyz))
    return xyz.astype(np.float64), np.asarray(opacity, dtype=np.float64)


def map_subset_indices(full_xyz, subset_xyz):
    distances, indices = cKDTree(full_xyz).query(subset_xyz, k=1)
    if len(distances) and float(distances.max()) > 1e-6:
        raise ValueError("labeled Gaussian PLY does not match the full model")
    return indices


def build_radius_neighbors(query_points, reference_tree, radius):
    """Store variable-length radius queries in a compact CSR-like tuple."""
    lists = reference_tree.query_ball_point(query_points, r=radius, workers=-1)
    counts = np.asarray([len(item) for item in lists], dtype=np.int64)
    indptr = np.zeros(len(query_points) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    non_empty = [np.asarray(item, dtype=np.int32) for item in lists if item]
    indices = (np.concatenate(non_empty) if non_empty
               else np.empty(0, dtype=np.int32))
    distances = np.empty(len(indices), dtype=np.float32)
    position = 0
    for query, neighbors in enumerate(lists):
        count = len(neighbors)
        if count:
            values = np.asarray(neighbors, dtype=np.int32)
            distances[position:position + count] = np.linalg.norm(
                reference_tree.data[values] - query_points[query], axis=1,
            )
        position += count
    return indptr, indices, distances


def save_neighbors(path, csr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indptr=csr[0], indices=csr[1], dist=csr[2])


def load_neighbors(path):
    data = np.load(path)
    return data["indptr"], data["indices"], data["dist"]
