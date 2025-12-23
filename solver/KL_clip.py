import math
import torch
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import safe_interact

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def clip_kl(gaussians, update, kl_threshold, lr):
    """
    Clip the update based on KL divergence threshold.

    Parameters:
    kl_threshold (float): The KL divergence threshold.
    """

    debug_clip = False

    # First find the total weight of each Gaussian and the per-channel weight
    # as opacity * features_dc

    # In rasterization, the dc color is shifted by +0.5 and clamped below to 0.
    SH0 = 0.28
    features_dc_min = -0.5
    opacity = gaussians.get_opacity
    features_dc = gaussians.get_features_dc
    C = (SH0 * features_dc - features_dc_min).clamp(min=1e-10)
    weights = (opacity * C.sum(dim=-1)).clip(min=1.0)

    ################# Position clip #################
    # We want \delta_x^T * covar^{-1} * \delta_x * weights < 2 * (KL_threshold / 3) for each axis
    xyz_delta = update.xyz
    quat = gaussians._rotation
    scaling = gaussians.get_scaling

    covar = build_covariance_from_scaling_rotation(
        scaling,
        1.0,
        quat
    )

    covar_diag = covar.diagonal(dim1=1, dim2=2)
    xyz_clip_thresh = torch.sqrt(2 * (kl_threshold / 3) * covar_diag / (weights + 1e-10))
    update.xyz.clip_(min=-xyz_clip_thresh, max=xyz_clip_thresh)


    ## Check ##
    if debug_clip:
        divergence_xyz = update.xyz * update.xyz / (covar_diag + 1e-10) * weights
        exceed_bound = divergence_xyz > 2 * (kl_threshold / 3)
        print(f"[KL Clip] Position clip: {exceed_bound.sum().item()} out of {exceed_bound.numel()} entries exceeded the bound.")


    # DEBUG: clip by lr
    update.xyz.clip_(min=-lr.xyz, max=lr.xyz)

    ################# and Rotation clip #################
    # We want (trace(covar^{-1} * updated_covar) - 3) * weights < 2 * KL_threshold
    quat = gaussians._rotation
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    updated_quat = gaussians._rotation + update.rotation
    updated_quat = updated_quat / torch.norm(updated_quat, dim=-1, keepdim=True)

    updated_covar = build_covariance_from_scaling_rotation(
        scaling,
        1.0,
        updated_quat
    )

    covar.diagonal(dim1=1, dim2=2).add_(1e-10)  # Numerical stability
    covar_delta = torch.linalg.solve(covar, updated_covar)
    traces = torch.diagonal(covar_delta, dim1=1, dim2=2).sum(dim=-1, keepdim=True) - 3
    traces_new = traces.clip(min=torch.zeros_like(weights), max=kl_threshold * 2 / (weights + 1e-10))
    update.rotation *= torch.sqrt(traces_new / (traces + 1e-15))    # TODO: Check if sqrt is needed

    ## Check ##
    if debug_clip:
        quat = gaussians._rotation
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        updated_quat = gaussians._rotation + update.rotation
        updated_quat = updated_quat / torch.norm(updated_quat, dim=-1, keepdim=True)

        updated_covar = build_covariance_from_scaling_rotation(
            scaling,
            1.0,
            updated_quat
        )
        covar.diagonal(dim1=1, dim2=2).add_(1e-10)  # Numerical stability
        covar_delta = torch.linalg.solve(covar, updated_covar)
        traces = torch.diagonal(covar_delta, dim1=1, dim2=2).sum(dim=-1, keepdim=True) - 3
        exceed_bound = traces * weights > 2 * kl_threshold
        print(f"[KL Clip] Rotation clip: {exceed_bound.sum().item()} out of {exceed_bound.numel()} gaussians exceeded the bound.")

    # DEBUG: clip by lr
    update.rotation.clip_(min=-lr.rotation, max=lr.rotation)

    ################# Scaling clip #################
    # We want e^(2*delta_Sx) - delta_Sx - 1 < 2 * (KL_threshold / 3) / weights for each axis
    # Approximate this as e^(2*delta_Sx) < K + 1
    #                     delta_Sx > 0.5 * (-K - 1 + exp(-2K))
    # where K = 2 * (KL_threshold / 3) / weights
    K = 2 * (kl_threshold / 3) / (weights + 1e-10)
    scaling_thresh_max = 0.5 * torch.log(K + 1)
    scaling_thresh_min = 0.5 * (-K - 1 + torch.exp(-2 * K))
    update.scaling.clip_(min=scaling_thresh_min, max=scaling_thresh_max)

    ## Check ##
    if debug_clip:
        divergence = torch.exp(2 * update.scaling) - update.scaling - 1
        exceed_bound = divergence * weights > 2 * (kl_threshold / 3)
        print(f"[KL Clip] Scaling clip: {exceed_bound.sum().item()} out of {exceed_bound.numel()} entries exceeded the bound.")

    # DEBUG: clip by lr
    update.scaling.clip_(min=-lr.scaling, max=lr.scaling)

    ################# Feature DC clip #################
    # We want O * C * log(O * C / (O * (C + delta_C))) - O * C + O * (C + delta_C) < 2 * (KL_threshold / 3) per channel
    # Where O is the opacity
    # Simplify to log(C / (C + delta_C)) + (C + delta_C) / C - 1 < 2 * (KL_threshold / 3) / (O * C)
    # Let S = (C + delta_C) / C
    # Then -log(S) + S < K + 1, where K = 2 * (KL_threshold / 3) / O
    # Approximate as S < K + 1 + log(K + 1)
    #                log(S) > -(K + 1) + S => S > exp(-(K + 1) + 1)

    updated_C = SH0 * (features_dc + update.features_dc) - features_dc_min
    S = updated_C / C
    K = 2 * (kl_threshold / 3) / (opacity.unsqueeze(-1) * C + 1e-10)
    S_max = K + 1 + torch.log(K + 1)
    S_min = torch.exp(-(K + 1) + 1)
    new_S = S.clip(min=S_min, max=S_max)
    update.features_dc = (new_S * C + features_dc_min) / SH0 - features_dc

    ## Check ##
    if debug_clip:
        updated_C = SH0 * (features_dc + update.features_dc) - features_dc_min
        S = updated_C / C
        divergence = -torch.log(S) + S - 1
        exceed_bound = divergence * (opacity.unsqueeze(-1) * C) > 2 * (kl_threshold / 3)
        print(f"[KL Clip] Feature DC clip: {exceed_bound.sum().item()} out of {exceed_bound.numel()} entries exceeded the bound.")

    # DEBUG: clip by lr 
    update.features_dc.clip_(min=-lr.features_dc, max=lr.features_dc)

    ################# Features rest clip #################
    # Same logic as feature dc but with 15 channels
    # Still needs a shift because features cannot be negative
    SH_rest = 1.0       # The higher this is, the more aggressive the clipping
    features_rest_min = -0.5
    features_rest = gaussians.get_features_rest
    C_rest = (SH_rest * features_rest - features_rest_min).clamp(min=1e-10)
    updated_C_rest = SH_rest * (features_rest + update.features_rest) - features_rest_min
    S_rest = updated_C_rest / C_rest
    K_rest = 2 * (kl_threshold / 15) / (opacity.unsqueeze(-1) * C_rest + 1e-10)
    S_rest_max = K_rest + 1 + torch.log(K_rest + 1)
    S_rest_min = torch.exp(-(K_rest + 1) + 1)
    new_S_rest = S_rest.clip(min=S_rest_min, max=S_rest_max)
    update.features_rest = (new_S_rest * C_rest + features_rest_min) / SH_rest - features_rest

    ## Check ##
    if debug_clip:
        updated_C_rest = SH_rest * (features_rest + update.features_rest) - features_rest_min
        S_rest = updated_C_rest / C_rest
        divergence_rest = -torch.log(S_rest) + S_rest - 1
        exceed_bound = divergence_rest * (opacity.unsqueeze(-1) * C_rest) > 2 * (kl_threshold / 15)
        print(f"[KL Clip] Feature rest clip: {exceed_bound.sum().item()} out of {exceed_bound.numel()} entries exceeded the bound.")

    # DEBUG: clip by lr
    update.features_rest.clip_(min=-lr.features_rest, max=lr.features_rest)

    ################# Opacity clip #################
    # We want O * C * log(O * C / ((O + delta_O) * C)) - O * C + (O + delta_O) * C < 2 * KL_threshold
    # Simplify to log(O / (O + delta_O)) + (O + delta_O) / O - 1 < 2 * KL_threshold / (O * C)
    # Let S = (O + delta_O) / O
    # Then -log(S) + S < K + 1, where K = 2 * KL_threshold / (O * C)

    updated_opacity = gaussians.opacity_activation(gaussians._opacity + update.opacity)
    weights = (opacity * C.sum(dim=-1))
    updated_weights = (updated_opacity * C.sum(dim=-1))
    S = updated_opacity / (opacity + 1e-10)
    K = 2 * kl_threshold / (weights + 1e-10)
    S_max = K + 1 + torch.log(K + 1)
    S_min = torch.exp(-(K + 1) + 1)
    new_S = S.clip(min=S_min, max=S_max)
    opacity_new = (new_S * opacity).clip(min=1e-5, max=1.0-1e-5)
    opacity.clip_(min=1e-5, max=1.0-1e-5)
    update.opacity = gaussians.inverse_opacity_activation(opacity_new) - gaussians.inverse_opacity_activation(opacity)
    # safe_interact(local=locals(), banner="after kl clip")

    ## Check ##
    if debug_clip:
        updated_opacity = gaussians.opacity_activation(gaussians._opacity + update.opacity)
        S = updated_opacity / (opacity + 1e-10)
        divergence = -torch.log(S) + S - 1
        exceed_bound = divergence * weights > 2 * kl_threshold
        print(f"[KL Clip] Opacity clip: {exceed_bound.sum().item()} out of {exceed_bound.numel()} entries exceeded the bound.")

    # DEBUG: clip by lr
    update.opacity.clip_(min=-lr.opacity, max=lr.opacity)

    # safe_interact(local=locals(), banner="after kl clip")

    return update


