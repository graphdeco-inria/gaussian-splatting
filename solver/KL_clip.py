import math
import torch
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import safe_interact

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def clip_kl(gaussians, update, kl_threshold, features_dc_lr, features_rest_lr, opacity_lr):
    """
    Clip the update based on KL divergence threshold.

    Parameters:
    kl_threshold (float): The KL divergence threshold.

    Should we clip by parameter or by parameter group?
    """

    # Position clip
    xyz_delta = update.xyz
    quat = gaussians._rotation
    scaling = gaussians.get_scaling

    covar = build_covariance_from_scaling_rotation(
        scaling,
        1.0,
        quat
    )


    # TODO: Check if actually need inverse covar
    covar_diag = covar.diagonal(dim1=1, dim2=2)
    xyz_clip_thresh = torch.sqrt(kl_threshold * covar_diag)

    # inverse_covar = covar.inverse()
    # inverse_covar_diag = inverse_covar.diagonal(dim1=1, dim2=2)
    # xyz_clip_thresh = torch.sqrt(kl_threshold / (inverse_covar_diag + 1e-15))

    update.xyz.clip_(min=-xyz_clip_thresh, max=xyz_clip_thresh)
    # xyz_update_norms = torch.einsum('bi,bij,bj->b', xyz_delta, inverse_covar, xyz_delta).abs()
    # xyz_update_norms_new = xyz_update_norms.clip(max=kl_threshold)
    # update.xyz *= torch.sqrt(xyz_update_norms_new / (xyz_update_norms + 1e-15))[:, None]

    # Rotation clip. TODO: Fix this
    quat = gaussians._rotation
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    updated_quat = gaussians._rotation + update.rotation
    updated_quat = updated_quat / torch.norm(updated_quat, dim=-1, keepdim=True)

    updated_covar = build_covariance_from_scaling_rotation(
        scaling,
        1.0,
        updated_quat
    )

    covar.diagonal(dim1=1, dim2=2).add_(1e-15)  # Numerical stability
    covar_delta = torch.linalg.solve(covar, updated_covar)
    # covar_delta = inverse_covar @ updated_covar
    traces = torch.diagonal(covar_delta, dim1=1, dim2=2).sum(dim=-1, keepdim=True) - 3
    traces_new = traces.clip(min=0, max=kl_threshold)
    update.rotation *= torch.sqrt(traces_new / (traces + 1e-15))    # TODO: Check if sqrt is needed

    # Scaling clip 
    scaling_thresh_min = 0.5 * (-kl_threshold - 1 + math.exp(-2 * kl_threshold))
    update.scaling.clip_(min=scaling_thresh_min, max=0.5*math.log(kl_threshold+1))
    # scaling_norms = torch.sum(update.scaling, dim=-1, keepdim=True)
    # scaling_norms_new = scaling_norms.clip(min=-kl_threshold-3, max=0.5*math.log(kl_threshold+3))
    # update.scaling *= scaling_norms_new / (scaling_norms + 1e-15)

    # Feature clip
    update.features_dc.clip_(min=-features_dc_lr, max=features_dc_lr)
    update.features_rest.clip_(min=-features_rest_lr, max=features_rest_lr)

    # Opacity clip. TODO: Verify how to do this
    update.opacity.clip_(min=-opacity_lr, max=opacity_lr)

    return update


