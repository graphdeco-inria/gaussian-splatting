import math
import torch
from utils.general_utils import safe_interact

def clip_uniform(gaussians, update, threshold, opacity_threshold, lr, quat_norm_tr=0.01):
    update.xyz.clamp_(-lr.xyz, lr.xyz)
    update.scaling.clamp_(-lr.scaling, lr.scaling)
    update.rotation.clamp_(-lr.rotation, lr.rotation)
    update.features_dc.clamp_(-lr.features_dc, lr.features_dc)
    update.features_rest.clamp_(-lr.features_rest, lr.features_rest)
    update.opacity.clamp_(-lr.opacity, lr.opacity)

    return update
