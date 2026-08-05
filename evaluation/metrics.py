"""IoU and panoptic metrics shared by Replica and ScanNet++."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from . import transfer


def class_iou(predicted, ground_truth, mask, class_id):
    """Calculate one-vs-rest IoU counts on the selected vertex mask.

    The return value is a dictionary containing true positives, false
    positives, false negatives, positive counts and the resulting IoU.
    """
    # Restrict both arrays to the visible annotated vertices used by evaluation.
    predicted_positive = (predicted[mask] == class_id)
    ground_truth_positive = (ground_truth[mask] == class_id)

    # Count one-vs-rest positives before calculating the union and IoU.
    tp = int((predicted_positive & ground_truth_positive).sum())
    fp = int((predicted_positive & ~ground_truth_positive).sum())
    fn = int((~predicted_positive & ground_truth_positive).sum())
    union = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": tp + fn,
        "pred_count": tp + fp,
        "iou": float(tp / union) if union else 0.0,
    }


def connected_components(xyz, radius, min_size):
    """Return radius-graph component labels and discard small components.

    The input contains one 3D position per predicted Gaussian. The returned
    array uses one component number per accepted Gaussian and -1 for discarded
    noise.
    """
    # Start every Gaussian as discarded noise until its component is accepted.
    output = np.full(len(xyz), -1, dtype=np.int64)
    if len(xyz) == 0:
        return output
    # The parent array implements a small union-find structure for the radius graph.
    parent = list(range(len(xyz)))

    def find(index):
        """Find the current root of one component in the union-find table."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # Connect every pair of Gaussians whose centers are within the radius.
    for left, right in cKDTree(xyz).query_pairs(radius):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    # Count the final component sizes before assigning compact IDs.
    roots = np.asarray([find(index) for index in range(len(xyz))])
    unique, counts = np.unique(roots, return_counts=True)
    accepted = {root for root, count in zip(unique, counts)
                if count >= min_size}
    # Remap accepted roots to compact IDs used by later instance matching.
    remapped = {}
    next_id = 0
    for index, root in enumerate(roots):
        if root not in accepted:
            continue
        if root not in remapped:
            remapped[root] = next_id
            next_id += 1
        output[index] = remapped[root]
    return output


def assign_vertex_instances(vertex_xyz, predicted_mask, gaussian_xyz,
                             gaussian_weights, gaussian_components, k, tau):
    """Assign each predicted vertex to its strongest nearby Gaussian.

    The arrays describe mesh positions, the vertices belonging to the target
    class, Gaussian positions and their weights and component labels. Only up
    to ``k`` nearby Gaussians inside the radius are considered.
    """
    # Vertices without an accepted nearby Gaussian keep the invalid label -1.
    output = np.full(len(vertex_xyz), -1, dtype=np.int64)
    query_indices = np.where(predicted_mask)[0]
    if len(query_indices) == 0 or len(gaussian_xyz) == 0:
        return output

    # Query only the vertices predicted as belonging to the target class.
    tree = cKDTree(gaussian_xyz)
    k_eff = min(k, len(gaussian_xyz))
    distances, indices = tree.query(vertex_xyz[query_indices], k=k_eff)
    if k_eff == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    # Nearby and opaque Gaussians receive the strongest vote.
    weights = gaussian_weights[indices] / (distances ** 2 + transfer.EPS)
    weights[distances > tau] = 0.0
    weights[gaussian_components[indices] < 0] = 0.0
    # Select the strongest valid component for each queried vertex.
    best = np.argmax(weights, axis=1)
    has_neighbor = weights[np.arange(len(query_indices)), best] > 0
    chosen = indices[np.arange(len(query_indices)), best]
    components = gaussian_components[chosen]
    accepted = has_neighbor & (components >= 0)
    output[query_indices[accepted]] = components[accepted]
    return output


def panoptic_quality(predicted_instances, ground_truth_instances, match_iou):
    """Return SQ, RQ and PQ for one class on one common vertex set.

    The two inputs contain one instance label per evaluated vertex. The return
    value is a dictionary with quality, recognition, matching and count
    information. Classes without matches keep SQ empty while PQ and RQ still
    account for missed instances.
    """
    # Ignore the invalid label and compare only actual predicted/GT instances.
    predicted_ids = [int(value) for value in np.unique(predicted_instances)
                     if value >= 0]
    ground_truth_ids = [int(value) for value in np.unique(ground_truth_instances)
                        if value >= 0]
    if not predicted_ids and not ground_truth_ids:
        return {
            "sq": None, "rq": None, "pq": None, "matches": 0,
            "n_pred": 0, "n_gt": 0, "matched_ious": [],
        }
    if not predicted_ids or not ground_truth_ids:
        return {
            "sq": None, "rq": 0.0, "pq": 0.0, "matches": 0,
            "n_pred": len(predicted_ids), "n_gt": len(ground_truth_ids),
            "matched_ious": [],
        }

    # Build the pairwise IoU matrix used by the Hungarian matching step.
    matrix = np.zeros((len(predicted_ids), len(ground_truth_ids)), dtype=float)
    for row, predicted_id in enumerate(predicted_ids):
        predicted_mask = predicted_instances == predicted_id
        for column, ground_truth_id in enumerate(ground_truth_ids):
            ground_truth_mask = ground_truth_instances == ground_truth_id
            intersection = int((predicted_mask & ground_truth_mask).sum())
            if intersection:
                matrix[row, column] = intersection / int(
                    (predicted_mask | ground_truth_mask).sum())

    # Prefer valid matches first, then maximize their IoU. Running Hungarian
    # on the raw matrix and filtering afterwards can discard a valid pair when
    # a different high-IoU assignment blocks more matches.
    # Invalid pairs receive a very high cost; valid pairs maximize their IoU.
    cost = np.where(matrix >= match_iou, -matrix, 1e6)
    rows, columns = linear_sum_assignment(cost)
    matched = [(row, column, float(matrix[row, column]))
               for row, column in zip(rows, columns)
               if matrix[row, column] >= match_iou]
    matches = len(matched)
    sq = (float(np.mean([item[2] for item in matched]))
          if matches else None)
    # Unmatched instances count as recognition errors in RQ and PQ.
    false_positive = len(predicted_ids) - matches
    false_negative = len(ground_truth_ids) - matches
    denominator = matches + 0.5 * false_positive + 0.5 * false_negative
    rq = float(matches / denominator) if denominator else 0.0
    pq = float((sq if sq is not None else 0.0) * rq)
    return {
        "sq": sq,
        "rq": rq,
        "pq": pq,
        "matches": matches,
        "n_pred": len(predicted_ids),
        "n_gt": len(ground_truth_ids),
        "matched_ious": [item[2] for item in matched],
    }


def _instance_metrics(scene, class_id, eval_mask, predicted_labels,
                      gaussian_xyz, gaussian_opacity, gaussian_components,
                      k, tau, match_iou):
    """Build predicted and GT instance labels before calculating PQ."""
    # Build predicted and reference instance maps on the same evaluated vertices.
    predicted_mask = (predicted_labels == class_id) & eval_mask
    predicted_instances = assign_vertex_instances(
        scene.vertices, predicted_mask, gaussian_xyz, gaussian_opacity,
        gaussian_components, k, tau,
    )
    ground_truth_instances = np.where(
        eval_mask & (scene.semantic_labels == class_id),
        scene.instance_labels,
        -1,
    )
    return panoptic_quality(predicted_instances, ground_truth_instances,
                            match_iou)


def evaluate_class(scene, cache, full_xyz, full_opacity, spec, predicted_xyz,
                   predicted_opacity, tau, min_share, opacity_weighted,
                   min_opacity, background_competes, k, pq_radius,
                   pq_min_component, pq_match_iou, ceiling_result=None):
    """Evaluate one target class and its representation ceiling.

    ``predicted_xyz`` and ``predicted_opacity`` may be absent when calculating
    the ceiling or an empty prediction. The boolean flags control opacity
    weighting and background competition during the mesh transfer. The cache
    supplies the precomputed neighborhoods and clean GT Gaussian labels.
    ``ceiling_result`` can reuse a ceiling already calculated for the class.

    The return value is a dictionary with IoU, ceiling IoU, panoptic metrics
    and the number of GT instances.
    """
    # Convert the canonical class name into the integer used by semantic arrays.
    class_id = scene.class_id(spec.name)
    eval_mask = scene.evaluation_mask
    gt_mask = eval_mask & (scene.semantic_labels == class_id)
    gt_instance_count = len(np.unique(scene.instance_labels[gt_mask]))

    if predicted_xyz is None or len(predicted_xyz) == 0:
        # Evaluate an empty prediction without trying to map a missing PLY file.
        predicted_labels = np.full(len(full_xyz), -1, dtype=np.int64)
        prediction_iou = class_iou(
            np.full(len(scene.vertices), -1, dtype=np.int64),
            scene.semantic_labels, eval_mask, class_id,
        )
        prediction_pq = panoptic_quality(
            np.full(len(scene.vertices), -1, dtype=np.int64),
            np.where(gt_mask, scene.instance_labels, -1),
            pq_match_iou,
        )
    else:
        # Map the labeled subset back into the full Gaussian model order.
        indices = transfer.map_subset_indices(full_xyz, predicted_xyz)
        predicted_labels = np.full(len(full_xyz), -1, dtype=np.int64)
        predicted_labels[indices] = class_id
        # Transfer predicted Gaussian labels onto the mesh using radius votes.
        vertex_labels = transfer.predict_vertex_labels(
            scene.vertices, cache.mesh_to_gaussian, predicted_labels,
            full_opacity, tau, min_share, opacity_weighted, min_opacity,
            background_competes,
        )
        # Build predicted instance components for the panoptic metrics.
        components = connected_components(predicted_xyz, pq_radius,
                                          pq_min_component)
        prediction_iou = class_iou(vertex_labels, scene.semantic_labels,
                                   eval_mask, class_id)
        prediction_pq = _instance_metrics(
            scene, class_id, eval_mask, vertex_labels, predicted_xyz,
            np.clip(predicted_opacity, min_opacity, 1.0), components, k, tau,
            pq_match_iou,
        )

    if ceiling_result is None:
        # The ceiling uses clean GT Gaussian labels instead of predicted labels.
        ceiling_mask = ((cache.gaussian_labels == class_id) &
                        (cache.gaussian_instances >= 0))
        ceiling_xyz = full_xyz[ceiling_mask]
        ceiling_opacity = full_opacity[ceiling_mask]
        ceiling_class_labels = np.where(
            ceiling_mask, class_id, -1,
        )
        ceiling_labels = transfer.predict_vertex_labels(
            scene.vertices, cache.mesh_to_gaussian, ceiling_class_labels,
            full_opacity, tau, min_share, opacity_weighted, min_opacity,
            background_competes,
        )
        ceiling_instances = cache.gaussian_instances[ceiling_mask]
        if len(ceiling_instances):
            # Compress original mesh instance IDs into local component IDs.
            _, ceiling_components = np.unique(ceiling_instances,
                                              return_inverse=True)
            ceiling_result = {
                "iou": class_iou(ceiling_labels, scene.semantic_labels,
                                  eval_mask, class_id),
                "pq": _instance_metrics(
                    scene, class_id, eval_mask, ceiling_labels, ceiling_xyz,
                    ceiling_opacity, ceiling_components, k, tau, pq_match_iou,
                ),
            }
        else:
            # A class without GT Gaussians has an empty representation ceiling.
            ceiling_result = {
                "iou": class_iou(
                    np.full(len(scene.vertices), -1, dtype=np.int64),
                    scene.semantic_labels, eval_mask, class_id,
                ),
                "pq": panoptic_quality(
                    np.full(len(scene.vertices), -1, dtype=np.int64),
                    np.where(gt_mask, scene.instance_labels, -1),
                    pq_match_iou,
                ),
            }

    return {
        "class": spec.name,
        "name_by_detector": spec.name_by_detector,
        "gt_instances": int(gt_instance_count),
        "iou": prediction_iou,
        "ceiling_iou": ceiling_result["iou"],
        "pq": prediction_pq,
        "ceiling_pq": ceiling_result["pq"],
    }


def _mean(values):
    """Return the mean of a non-empty collection or zero otherwise."""
    return float(np.mean(values)) if values else 0.0


def aggregate(per_class):
    """Aggregate classes with GT or evaluated predicted positives.

    A class absent from GT still contributes when the prediction produces
    positives on the evaluated mesh, so false-positive-only classes cannot be
    silently discarded. The return value contains the headline metrics and
    the class names included in each aggregate.
    """
    # Exclude classes with neither GT instances nor evaluated predictions.
    evaluated = [(name, item) for name, item in per_class.items()
                 if (item["gt_instances"] > 0 or
                     item["iou"]["pred_count"] > 0)]
    # Relative IoU is defined only when the class has a non-zero ceiling.
    relative = [item["iou"]["iou"] / item["ceiling_iou"]["iou"]
                for _, item in evaluated if item["ceiling_iou"]["iou"] > 0]
    relative_classes = [name for name, item in evaluated
                        if item["ceiling_iou"]["iou"] > 0]

    # Compute global IoU from pooled vertex counts rather than averaging IoUs.
    tp = sum(item["iou"]["tp"] for _, item in evaluated)
    fp = sum(item["iou"]["fp"] for _, item in evaluated)
    fn = sum(item["iou"]["fn"] for _, item in evaluated)
    union = tp + fp + fn

    # SQ is averaged only for classes with at least one valid instance match.
    pq_values = [item["pq"]["pq"] for _, item in evaluated]
    rq_values = [item["pq"]["rq"] for _, item in evaluated]
    sq_values = [item["pq"]["sq"] for _, item in evaluated
                 if item["pq"]["sq"] is not None]
    ceiling_iou_values = [item["ceiling_iou"]["iou"] for _, item in evaluated
                          if item["ceiling_iou"]["iou"] > 0]
    ceiling_pq_values = [item["ceiling_pq"]["pq"] for _, item in evaluated
                         if item["ceiling_pq"]["pq"] is not None
                         and item["ceiling_pq"]["pq"] > 0]

    return {
        "mIoU": _mean([item["iou"]["iou"] for _, item in evaluated]),
        "ceiling_mIoU": _mean(ceiling_iou_values),
        "relative_mIoU": _mean(relative),
        "global_iou": float(tp / union) if union else 0.0,
        "mPQ": _mean(pq_values),
        "mSQ": float(np.mean(sq_values)) if sq_values else None,
        "mRQ": _mean(rq_values),
        "ceiling_mPQ": _mean(ceiling_pq_values),
        "evaluated_classes": [name for name, _ in evaluated],
        "relative_classes": relative_classes,
        "mSQ_classes": [name for name, item in evaluated
                         if item["pq"]["sq"] is not None],
    }
