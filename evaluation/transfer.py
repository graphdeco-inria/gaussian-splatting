# Radius-vote transfers between mesh vertices and Gaussians

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

EPS = 1e-10


def sigmoid(values):
    """ Convert opacity into values between zero and one """

    # Gaussian PLY files store opacity before the sigmoid conversion
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def load_gaussian_ply(path):
    """
    Load Gaussian centers and opacity values converted to alpha

    The return value contains one array of 3D centers and one array of opacity values in the same Gaussian order
    """

    # Keep the original Gaussian order because the labels use local IDs aligned with the original Gaussian rows.
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
    names = vertex.data.dtype.names or ()

    if "opacity" in names:
        # Convert stored opacity into alpha values used as vote weights
        opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))

    else:
        # Older PLY files without opacity give every Gaussian equal weight
        opacity = np.ones(len(xyz), dtype=np.float64)
    return xyz, opacity


def map_subset_indices(full_xyz, subset_xyz):
    """ Map a subset of Gaussian centers, the predicted ones, back to their positions in the full Gaussian representation """

    # Labeled PLY files contain a subset of the full model in a new order
    distance, indices = cKDTree(full_xyz).query(subset_xyz, k=1)
    if float(distance.max()) > 1e-6:
        raise ValueError("labeled Gaussian PLY does not match the full model")
    return indices


def build_radius_neighbors(query_points, reference_tree, radius, chunk_size=100000):
    """
    Build radius neighborhoods (points from reference_tree in a radius) for every query point

    The result contains row offsets, neighbor indices and distances
    Rows correspond to query points and the indices refer to points in the search tree
    """

    # Store each chunk separately so large scenes do not require one huge query
    index_chunks = []
    distance_chunks = []
    count_chunks = []

    # Query the reference KD-tree in chunks
    for start in range(0, len(query_points), chunk_size):
        end = min(start + chunk_size, len(query_points))

        # Query all points in this chunk against the reference KD-tree
        lists = reference_tree.query_ball_point(query_points[start:end], r=radius)

        # CSR row counts allow the variable length neighbor lists to be flattened
        counts = np.fromiter((len(item) for item in lists), dtype=np.int64, count=len(lists))
        count_chunks.append(counts) # Counts the length of each neighbor list in this query vertex chunk
        chunk_indices = np.empty(int(counts.sum()), dtype=np.int32)
        chunk_distances = np.empty(int(counts.sum()), dtype=np.float32)
        position = 0

        # For every query point in this chunk, store the neighbor indices and distances in flat arrays
        for query_index, neighbors in enumerate(lists):
            count = len(neighbors)

            if count:
                # Save neighbor indices and their distances in flat arrays
                neighbor_indices = np.asarray(neighbors, dtype=np.int32)
                chunk_indices[position:position + count] = neighbor_indices
                chunk_distances[position:position + count] = np.linalg.norm(
                    reference_tree.data[neighbor_indices] - query_points[start + query_index], axis=1)
            position += count
        index_chunks.append(chunk_indices)
        distance_chunks.append(chunk_distances)

    # Join all chunks and build the CSR array
    counts = np.concatenate(count_chunks) if count_chunks else np.empty(0, dtype=np.int64)
    indptr = np.zeros(len(query_points) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.concatenate(index_chunks) if index_chunks else np.empty(0, dtype=np.int32)
    distances = (np.concatenate(distance_chunks) if distance_chunks else np.empty(0, dtype=np.float32))
    return indptr, indices, distances


def save_neighbors(path, csr):
    """ Save a neighborhood structure"""

    # Store the three CSR components for later reuse
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indptr=csr[0], indices=csr[1], dist=csr[2])


def load_neighbors(path):
    """ Load row offsets, neighbor indices and distances from disk """

    # Return the same tuple shape produced by build_radius_neighbors
    data = np.load(path)
    return data["indptr"], data["indices"], data["dist"]


def radius_label_vote(n_query, csr, reference_labels, reference_weights, classes, min_share, background_labels_compete):
    """
    Assign local labels to a vertex with a weighted radius vote from Gaussians and optional abstention

     This function works in both directions, from mesh vertices to Gaussians and from Gaussians to mesh vertices
    When Gaussians to mesh, the reference labels are the local IDs of the Gaussians and the query points are mesh vertices

    More precisely, for each query point (mesh vertex), we have a set of neighboring reference points (Gaussians) with known local labels and weights
    We want to assign a local label to each query point based on the weighted votes of its neighbors

    n_query is the number of query points
    csr contains one neighborhood row per query point 
    Reference labels and weights use the order of the referenced points
    background_labels_compete decides whether non-target labels, -1, contribute to the vote denominator.
    """

    # Expand the CSR structure
    indptr, indices, distances = csr
    if len(indices) == 0: # If there are no neighbors, return an array of invalid labels for all query points
        return np.full(n_query, -1, dtype=np.int64)

    '''
    Suppose:

    indptr = [0, 3, 5]
    indices = [3, 7, 10, 2, 4]
    distances = [0.01, 0.02, 0.04, 0.03, 0.01]
    
    Then:
        Query 0:
            Reference 3  → 0.01
            Reference 7  → 0.02
            Reference 10 → 0.04

        Query 1:
            Reference 2  → 0.03
            Reference 4  → 0.01

    indptr = [0, 3, 5] produces [3, 2], as
        Query 0 → 3 neighbors
        Query 1 → 2 neighbors

    And np.repeat produces [0, 0, 0, 1, 1], which is the query index for each edge.

    '''

     # Imagine one query vertex. There is then one edge per neighbor, which has its own label and weight. The query index is repeated for every edge of that query vertex.
    # These three arrays have the length of indices
    edge_query = np.repeat(np.arange(n_query, dtype=np.int64), np.diff(indptr))
    edge_labels = reference_labels[indices]
    edge_weights = (reference_weights[indices].astype(np.float64) / (distances.astype(np.float64) ** 2 + EPS)) # Closer neighbors have more influence

    '''
    Imagine:
      Edge   Query   Reference   Label   Weight 
                                                
         0       0           3       0      100 
         1       0           7       0       25 
         2       0          10       1     6.25 
         3       1           2       1    11.11 
         4       1           4       0      100 

    For query 0, the total weight is 100 + 25 + 6.25 = 131.25
    Then, total[0] = 131.25
    And repeats for every query vertex
    '''

    # The denominator either includes or excludes background labels
    if background_labels_compete:
        total = np.bincount(edge_query, weights=edge_weights, minlength=n_query) # Weighted sum per binning query vertex
    else:
        total = np.bincount(edge_query, weights=np.where(edge_labels >= 0, edge_weights, 0.0), minlength=n_query)

    '''
    For query 0: (one cell in the bincount)
        The score for label 0 is 100 + 25 = 125
        The score for label 1 is 6.25
        The share for label 0 is 125 / 131.25 = 0.952
        The share for label 1 is 6.25 / 131.25 = 0.048
        If min_share = 0.5, then label 0 is accepted and label 1 is rejected. The query vertex receives label 0
        If min_share = 0.99, then both labels are rejected and the query vertex receives -1
    '''

    # Start with no local class selected
    best_score = np.zeros(n_query, dtype=np.float64) # There is one competition for each query vertex
    best_label = np.full(n_query, -1, dtype=np.int64)

    # We initially consider the background label as the best score, so that target labels must beat it to be accepted
    if background_labels_compete:
        best_score = np.bincount(
            edge_query,
            weights=np.where(edge_labels < 0, edge_weights, 0.0),
            minlength=n_query,
        )

    # Compare the weighted score of every target main-local class for every query point.
    for label in classes:
        score = np.bincount(
            edge_query,
            weights=np.where(edge_labels == label, edge_weights, 0.0),
            minlength=n_query,
        )
        better = score > best_score
        best_score[better] = score[better]
        best_label[better] = label

    # Abstain when the winning class does not reach the required vote share
    share = np.divide(best_score, total, out=np.zeros_like(best_score), where=total > 0)
    accepted = ((total > 0) & (best_score > 0) & (share >= min_share))
    output = np.full(n_query, -1, dtype=np.int64)
    output[accepted] = best_label[accepted]
    return output

def nearest_neighbor_label(n_query, csr, reference_labels, tau):
    """ Assign each query point the label of its nearest reference point """

    # Expand the CSR structure
    indptr, indices, distances = csr
    output = np.full(n_query, -1, dtype=np.int64)

    # For every query point, find the nearest reference point and assign its label if it is within the radius tau
    for query_index in range(n_query):
        start, end = int(indptr[query_index]), int(indptr[query_index + 1])

        if start == end:
            continue
        nearest_offset = start + int(np.argmin(distances[start:end]))

        if distances[nearest_offset] <= tau:
            output[query_index] = reference_labels[indices[nearest_offset]]
    return output


def predict_vertex_labels(mesh_xyz, mesh_gaussian_csr, gaussian_labels, gaussian_opacity, tau, min_share,
                          opacity_weighted, min_opacity, gaussian_to_mesh_background_competes,
                          gaussian_to_mesh_transfer):
    """
    Transfer Gaussian local labels to mesh vertices with the selected method

    opacity_weighted decides whether Gaussian opacity affects the vote weights
    gaussian_to_mesh_background_competes decides whether background neighbors compete with target labels
    gaussian_to_mesh_transfer selects radius voting or nearest-neighbor assignment
    Vertices without an accepted label receive the invalid label value
    """

    if gaussian_to_mesh_transfer == "nearest_neighbor_label":
        return nearest_neighbor_label(len(mesh_xyz), mesh_gaussian_csr, gaussian_labels, tau)

    # Select opacity weight or uniform weights before running the vote
    if opacity_weighted: # As we compute the score for several classes, we use as weight the opacity of the Gaussian, which does not depend on the class
        weights = np.clip(gaussian_opacity, min_opacity, 1.0)
    else:
        weights = np.ones(len(gaussian_labels), dtype=np.float64)

    # Only valid local labels can participate as target classes
    classes = np.unique(gaussian_labels[gaussian_labels >= 0])
    return radius_label_vote(len(mesh_xyz), mesh_gaussian_csr, gaussian_labels, weights, classes, min_share, gaussian_to_mesh_background_competes)
