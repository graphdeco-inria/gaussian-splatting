"""Mesh IoU and connected components used by the Replica prototype."""

import numpy as np
from scipy.spatial import cKDTree


def class_iou(predicted, ground_truth, mask, class_id):
    predicted_positive = predicted[mask] == class_id
    ground_truth_positive = ground_truth[mask] == class_id
    tp = int((predicted_positive & ground_truth_positive).sum())
    fp = int((predicted_positive & ~ground_truth_positive).sum())
    fn = int((~predicted_positive & ground_truth_positive).sum())
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "gt_count": tp + fn,
            "pred_count": tp + fp,
            "iou": float(tp / union) if union else 0.0}


def connected_components(xyz, radius, min_size):
    """Keep Gaussian components large enough to represent an object."""
    labels = np.full(len(xyz), -1, dtype=np.int64)
    if not len(xyz):
        return labels
    parent = list(range(len(xyz)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in cKDTree(xyz).query_pairs(radius):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root
    roots = np.asarray([find(index) for index in range(len(xyz))])
    unique, counts = np.unique(roots, return_counts=True)
    accepted = {root for root, count in zip(unique, counts) if count >= min_size}
    remapped = {}
    for index, root in enumerate(roots):
        if root in accepted:
            remapped.setdefault(root, len(remapped))
            labels[index] = remapped[root]
    return labels
