""" Vertex IoU used while moving the evaluation out of the old workflow """

import numpy as np


def class_iou(predicted, ground_truth, mask, class_id):
    """Calculate one-vs-rest counts on the selected mesh vertices."""
    predicted_positive = predicted[mask] == class_id
    ground_truth_positive = ground_truth[mask] == class_id
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
