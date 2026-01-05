import math
import torch
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import safe_interact

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def clip_kl(gaussians, update, kl_threshold, gamma, lr):
    """
    Split the target KL divergence into mass and shape components.
    First solve the mass component, then use the parameters to solve the shape component.


    A Gaussian splat is (2 pi)^(3/2) * opacity * color * det(Sigma)^(1/2) * N(mu, Sigma), because the Gaussian split is unnormalized.
    We can omit the (2 pi)^(3/2) factor.

    det(Sigma)^(1/2) = scaling_x * scaling_y * scaling_z
    We will approximate color by (SH0 * features_dc + 0.5)
    We will let features_rest be clipped by lr
    """

    SH0 = 0.28
    features_dc_min = -0.5

    epsilon_mass = kl_threshold * (1 - gamma)
    epsilon_shape = kl_threshold * gamma

    xyz = gaussians.get_xyz
    quat = gaussians.get_rotation
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    features_dc = gaussians.get_features_dc
    C = (SH0 * features_dc - features_dc_min).clamp(min=1e-10)

    det_sqrt = scaling[:, 0:1] * scaling[:, 1:2] * scaling[:, 2:3]
    M_per_channel = opacity.unsqueeze(-1) # * C * det_sqrt.unsqueeze(-1)
    M = M_per_channel.max(dim=-1).values

    ################# Color clip #################
    """
    S_color = (opacity * color_new * det_sqrt) / (opacity * color * det_sqrt) = color_new / color
    S_color_min < S_color < S_color_max
    1 - kappa < S_color < 1 + kappa
    """
    C_new = (SH0 * (features_dc + update.features_dc) - features_dc_min).clamp(min=1e-10)
    S_color = C_new / C
    kappa_color = (epsilon_mass / (3 * M_per_channel + 1e-10)).sqrt()
    S_color.clip_(min=1-kappa_color, max=1+kappa_color)
    C_new = S_color * C
    update.features_dc = (C_new + features_dc_min) / SH0 - features_dc

    # TODO: See if we want to clip features_rest as well
    update.features_rest.clip_(min=-lr.features_rest, max=lr.features_rest)

    ################# Opacity clip #################
    """
    S_opacity = (opacity_new * color * det_sqrt) / (opacity * color * det_sqrt) = opacity_new / opacity
    S_opacity_min < S_opacity < S_opacity_max
    1 - kappa < S_opacity < 1 + kappa
    """
    opacity_new = gaussians.opacity_activation(gaussians._opacity + update.opacity)
    S_opacity = opacity_new / (opacity + 1e-10)
    kappa_opacity = (epsilon_mass / (M + 1e-10)).sqrt()
    S_opacity.clip_(min=1-kappa_opacity, max=1+kappa_opacity)
    opacity_new = S_opacity * opacity
    update.opacity = gaussians.inverse_opacity_activation(opacity_new.clip(min=1e-5,max=1-1e-5)) - gaussians.inverse_opacity_activation(opacity.clip(min=1e-5,max=1-1e-5))

    ################# Scaling clip (Mass) #################
    """
    S_scaling_x = (opacity * color * exp(Sx_new + Sy + Sz)) / (opacity * color * exp(Sx + Sy + Sz)) = exp(delta S_x)
    S_scaling_x_min < S_scaling_x < S_scaling_x_max
    1 - kappa < S_scaling_x < 1 + kappa
    log(1 - kappa) < delta S_x < log(1 + kappa)
    """
    kappa_scaling = (epsilon_mass / (3 * M + 1e-10)).sqrt()
    scaling_thresh_min = torch.log((1 - kappa_scaling).clip(min=1e-10))
    scaling_thresh_max = torch.log(1 + kappa_scaling)
    update.scaling.clip_(min=scaling_thresh_min, max=scaling_thresh_max)


    """
    Shape clip:
    First update worst-case M_new after mass clip
    Then bound KL divergence by D_KL < epsilon_shape / M_new
    """
    scaling_new = gaussians.scaling_activation(gaussians._scaling + update.scaling.clip(min=0.0)) # Worst-case scaling update
    # opacity_new
    # color_new
    # det_sqrt_new = scaling_new[:, 0:1] * scaling_new[:, 1:2] * scaling_new[:, 2:3]
    # M_per_channel_new = opacity_new.unsqueeze(-1) * C_new * det_sqrt_new.unsqueeze(-1)
    # M_new = M_per_channel_new.max(dim=-1).values

    # DEBUG
    # epsilon_shape = 2 * epsilon_shape / (M_new + 1e-10)
    epsilon_shape = epsilon_shape * torch.ones_like(opacity)

    ################# Position clip #################
    covar = build_covariance_from_scaling_rotation(scaling, 1.0, quat)
    covar_diag = covar.diagonal(dim1=1, dim2=2).clamp(min=1e-10)
    kappa_xyz = ((epsilon_shape / 3) * covar_diag).sqrt()
    update.xyz.clip_(min=-kappa_xyz, max=kappa_xyz)

    ################# Rotation clip #################
    # """
    # First normalize rotation updates so that they are approximately linear
    # """
    # max_rotation_norm = 2 * lr.rotation
    # update.rotation *= (max_rotation_norm / (torch.norm(update.rotation, dim=-1, keepdim=True) + 1e-10)).clip(max=1.0)

    quat_new = gaussians.rotation_activation(gaussians._rotation + update.rotation)
    covar_new = build_covariance_from_scaling_rotation(scaling, 1.0, quat_new)
    covar.diagonal(dim1=1, dim2=2).add_(1e-10)  # Numerical stability
    covar_delta = torch.linalg.solve(covar, covar_new)
    traces = torch.diagonal(covar_delta, dim1=1, dim2=2).sum(dim=-1, keepdim=True) - 3
    traces_new = traces.clip(min=torch.zeros_like(traces), max=epsilon_shape)
    update.rotation *= torch.sqrt(traces_new / (traces + 1e-15))    # TODO: Check if sqrt is needed

    ################## Scaling clip (Shape) #################
    """
    e^(2 * delta Sx) - 2x - 1 < epsilon_shape / 3
    Approximate as |delta Sx| < sqrt(epsilon_shape / 3 / 2e)
    """

    kappa_scaling_shape = (epsilon_shape / (3 * 2 * math.e)).sqrt()
    update.scaling.clip_(min=-kappa_scaling_shape, max=kappa_scaling_shape)

    # # DEBUG: truncate position, rotation, scaling
    # update.xyz.clip_(min=-lr.xyz, max=lr.xyz)
    # update.rotation.clip_(min=-lr.rotation, max=lr.rotation)
    # update.scaling.clip_(min=-lr.scaling, max=lr.scaling)

    # DEBUG: truncate color and opacity
    update.features_dc.clip_(min=-lr.features_dc, max=lr.features_dc)
    update.features_rest.clip_(min=-lr.features_rest, max=lr.features_rest)
    # update.opacity.clip_(min=-lr.opacity, max=lr.opacity)

    # safe_interact(local=locals(), banner="after KL clip")

    return update

