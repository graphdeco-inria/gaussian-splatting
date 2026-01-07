import math
import torch
from utils.general_utils import safe_interact
from solver.gaussian_model_vector import GaussianModelVector
from diff_gaussian_rasterization import compute_trust_region_step

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=r.dtype, device=r.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def clip_hellinger(gaussians, update, threshold, lr, quat_norm_tr=0.01):
    """
    For two unnormalized gaussians denoted by P(x) = C * N(x; mu, Sigma) and Q(x) = C' * N(x; mu', Sigma'),
    the squared Hellinger distance is given by:
        H^2(P, Q) = 1/2 (C + C') - (C * C')^(1/2) * |Sigma|^(1/4) * |Sigma'|^(1/4) / |(Sigma + Sigma')/2|^(1/2) * exp(-1/8 (mu - mu')^T ((Sigma + Sigma')/2)^(-1) (mu - mu'))
    We want to clip updates such that H^2(P, Q) < threshold.
    Each update parameter introduces some contribution to H^2, and we will split the contributions
    between position, rotation, scaling, opacity, and color evenly.
    When computing the contributions, we will approximate the other parameters as constant.

    C is the mass of the gaussian splat, which is given by:
    opacity * color * det(Sigma)^(1/2)
    """

    # Gaussians with lower mass will be allowed to shift more
    # Clip this number because multiple parameters can update at once
    MIN_MASS_DENOM = 0.1
    MAX_MASS_DENOM = 10000.0

    xyz = gaussians.get_xyz                                             # (P, 3)   
    scaling = gaussians.get_scaling                                     # (P, 3)     
    quat = gaussians.get_rotation                                       # (P, 4)
    features_dc = gaussians.get_features_dc                             # (P, 1, 3)  
    features_rest = gaussians.get_features_rest                         # (P, 15, 3)
    opacity = gaussians.get_opacity                                     # (P, 1)

    clip_xyz_pytorch = True
    clip_scaling_pytorch = True
    clip_rotation_pytorch = True
    clip_opacity_pytorch = True
    clip_features_dc_pytorch = True
    clip_features_rest_pytorch = True

    if False:

        det_sqrt = scaling[:, 0:1] * scaling[:, 1:2] * scaling[:, 2:3]      # (P, 1)

        SH0 = 0.282
        SH_rest = 1.0
        features_dc_min = -0.5
        color = (SH0 * features_dc - features_dc_min).clamp(min=1e-20)      # (P, 1, 3)

        covar = build_covariance_from_scaling_rotation(scaling, 1.0, quat)  # (P, 3, 3)


        ################### Position clip #################
        """
        delta_x < sqrt(-8 * Sigma_xx * log(1 - epsilon / (opacity * color)))
        """

        epsilon_xyz = threshold / 3    # DEBUG
        epsilon_xyz = epsilon_xyz / (opacity).clamp(min=MIN_MASS_DENOM, max=MAX_MASS_DENOM)   # (P, 1)
        covar_xyz = covar + 1e-5 * torch.eye(3, device=covar.device).unsqueeze(0)  # (P, 3, 3)
        covar_inv = torch.linalg.inv(covar_xyz)                                 # (P, 3, 3)
        covar_inv_diag = torch.diagonal(covar_inv, dim1=-2, dim2=-1)                # (P, 3)
        xyz_thresh = torch.sqrt((-8 * (1 / covar_inv_diag) * torch.log((1 - epsilon_xyz).clamp(min=1e-20))).clamp(min=1e-20))  # (P, 3)
        if clip_xyz_pytorch:
            update.xyz.clip_(min=-xyz_thresh, max=xyz_thresh)

        # # DEBUG
        # for i in range(3):
        #     update_copy = update * 0.0
        #     update_copy.xyz[:, i] = update.xyz[:, i]
        #     H_squared = compute_hellinger_distance(gaussians, update_copy, scale_invariant=True, debug=False)
        #     param = ["x", "y", "z"][i]
        #     print(f"param {param} H_squared max: {H_squared.max().item():.6f}, mean: {H_squared.mean().item():.6f}")

        # update_copy = update * 0.0
        # update_copy.xyz = update.xyz
        # H_squared = compute_hellinger_distance(gaussians, update_copy, scale_invariant=True, debug=False)
        # print(f"param xyz combined H_squared max: {H_squared.max().item():.6f}, mean: {H_squared.mean().item():.6f}")


        ################### Rotation clip #################
        """
        tr(S^{-2} \Delta R^T S^2 \Delta R) < 3 - ln(1 - epsilon / (opacity * color))
        \|S^{-1} (I + G) S\|_F^2 - 3 < -ln(1 - epsilon / (opacity * color))
        """

        epsilon_rotation = threshold / 4
        epsilon_rotation = epsilon_rotation / (opacity).clamp(min=MIN_MASS_DENOM, max=MAX_MASS_DENOM)   # (P, 1)
        quat_coeffs = compute_quat_to_trace_coefficient(gaussians._rotation, scaling)  # (P, 4)
        quat_thresh_hellinger = torch.sqrt((-8 / quat_coeffs * torch.log((1 - epsilon_rotation).clamp(min=1e-20))).clamp(min=1e-20))
        update.rotation.clip_(min=-quat_thresh_hellinger, max=quat_thresh_hellinger)
        # quat_thresh = torch.sqrt(epsilon_rotation.clamp(min=1e-20))
        quat_norm_thresh = gaussians._rotation.norm(dim=-1, keepdim=True) * quat_norm_tr
        quat_thresh = torch.min(quat_norm_thresh, quat_thresh_hellinger)

        if clip_rotation_pytorch:
            update.rotation.clip_(min=-quat_thresh, max=quat_thresh)
        # safe_interact(local=locals(), banner="hellinger_clip_rotation_debug")


        # # DEBUG
        # for i in range(4):
        #     update_copy = update * 0.0
        #     update_copy.rotation[:, i] = update.rotation[:, i]
        #     H_squared = compute_hellinger_distance(gaussians, update_copy, scale_invariant=True, debug=False)
        #     param = ["qx", "qy", "qz", "qw"][i]
        #     print(f"param {param} H_squared max: {H_squared.max().item():.6f}, mean: {H_squared.mean().item():.6f}")

        # update_copy = update * 0.0
        # update_copy.rotation = update.rotation
        # H_squared = compute_hellinger_distance(gaussians, update_copy, scale_invariant=True, debug=False)
        # print(f"param rotation combined H_squared max: {H_squared.max().item():.6f}, mean: {H_squared.mean().item():.6f}")

        # safe_interact(local=locals(), banner="hellinger_clip_debug")

        ################### Scaling clip #################
        """
        delta_Sx < sqrt(4/3 * S_x^2 * epsilon / (opacity * color))
        """

        epsilon_scaling = threshold / 3
        epsiilon_scaling = epsilon_scaling / (opacity).clamp(min=MIN_MASS_DENOM, max=MAX_MASS_DENOM)   # (P, 1)
        scaling_thresh = torch.sqrt((4/3 * (scaling ** 2) * epsiilon_scaling).clamp(1e-20))            # (P, 3)

        # This scaling thresh is for bounding scaling after exponentiation
        scaling_new = gaussians.scaling_activation(gaussians._scaling + update.scaling).clamp(min=1e-20)
        scaling_new.clip_(min=scaling - scaling_thresh, max=scaling + scaling_thresh)
        update.scaling = gaussians.scaling_inverse_activation(scaling_new) - gaussians._scaling

        # ## DEBUG: Check ##
        # check_hellinger_scaling(gaussians, update, scale_invariant=True, debug=True)


        ################### Opacity clip #################
        """
        delta_alpha < sqrt(4 * opacity * epsilon / color)
        """
        mass_scale = 1.0 # 0.01

        epsilon_opacity = threshold * mass_scale
        opacity_thresh = torch.sqrt(4 * opacity * epsilon_opacity)        # (P, 1)

        # This opacity thresh is for bounding opacity after activation
        opacity_new = gaussians.opacity_activation(gaussians._opacity + update.opacity)
        opacity_new.clip_(min=opacity - opacity_thresh, max=opacity + opacity_thresh)
        opacity_new.clip_(min=1e-5, max=1.0-1e-5)
        update.opacity = gaussians.inverse_opacity_activation(opacity_new) - gaussians._opacity


        ################### Color clip #################
        """
        delta_color < sqrt(4 * color * epsilon / opacity)
        """

        epsilon_color = threshold / 3 * mass_scale
        epsilon_color = epsilon_color / (opacity).clamp(min=MIN_MASS_DENOM, max=MAX_MASS_DENOM) # (P, 1)
        color_thresh = torch.sqrt(4 * color * epsilon_color.unsqueeze(-1))  # (P, 1, 3)

        # This color thresh is for bounding color after activation
        color_new = (SH0 * (features_dc + update.features_dc) - features_dc_min).clamp(min=1e-20)
        color_new.clip_(min=color - color_thresh, max=color + color_thresh)
        update.features_dc = (color_new + features_dc_min) / SH0 - features_dc

        
        # epsilon_color_rest = threshold / 15 * mass_scale
        # epsilon_color_rest = epsilon_color_rest / (opacity).clamp(min=MIN_MASS_DENOM, max=MAX_MASS_DENOM) # (P, 1)
        # color_rest_thresh = torch.sqrt(4 * color * epsilon_color_rest.unsqueeze(-1))  # (P, 1, 3)

        # # This color thresh is for bounding color after activation
        # color_rest = (SH_rest * features_rest - features_dc_min).clamp(min=1e-20)
        # color_rest_new = (SH_rest * (features_rest + update.features_rest) - features_dc_min).clamp(min=1e-20)
        # color_rest_new.clip_(min=color_rest - color_rest_thresh, max=color_rest + color_rest_thresh)
        # update.features_rest = (color_rest_new + features_dc_min) / SH_rest - features_rest

        # update.xyz.clip_(min=-lr.xyz, max=lr.xyz)
        # update.scaling.clip_(min=-lr.scaling, max=lr.scaling)
        # update.rotation.clip_(min=-lr.rotation, max=lr.rotation)

        # update.opacity.clip_(min=-lr.opacity, max=lr.opacity)
        # update.features_dc.clip_(min=-lr.features_dc, max=lr.features_dc)
        update.features_rest.clip_(min=-lr.features_rest, max=lr.features_rest)
        # safe_interact(local=locals(), banner="hellinger_clip")

        # compute_hellinger_distance(gaussians, update, scale_invariant=True, debug=True)

    # update_copy = update.clone()

    xyz_step_clipped, scaling_step_clipped, quat_step_clipped, opacity_step_clipped, shs_step_clipped  = compute_trust_region_step(
            gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features,
            update.xyz, update.scaling, update.rotation, update.opacity, torch.cat([update.features_dc, update.features_rest], dim=1),
            threshold, MIN_MASS_DENOM, MAX_MASS_DENOM, scale_modifier=1.0, quat_norm_tr=quat_norm_tr,)

    update.xyz = xyz_step_clipped
    update.scaling = scaling_step_clipped
    update.rotation = quat_step_clipped
    update.opacity = opacity_step_clipped
    update.features_dc = shs_step_clipped[:, 0:1, :]
    update.features_rest = shs_step_clipped[:, 1:, :]

    # safe_interact(local=locals(), banner="hellinger_clip_final_debug")


    return update

def check_hellinger_scaling(gaussians, update, scale_invariant=True, debug=False):
    dtype = torch.double
    SH0 = 0.282
    SH_rest = 1.0
    color_rest_min = -0.5
    scale_modifier = 1.0

    xyz = gaussians.get_xyz.to(dtype)                                             # (P, 3)
    scaling = gaussians.scaling_activation(gaussians._scaling.to(dtype))          # (P, 3)
    quat = gaussians.rotation_activation(gaussians._rotation.to(dtype))           # (P, 4)
    features_dc = gaussians.get_features_dc.to(dtype)                             # (P, 1, 3)
    features_rest = gaussians.get_features_rest.to(dtype)                         # (P, 15, 3)
    opacity = gaussians.opacity_activation(gaussians._opacity.to(dtype))          # (P, 1)
    covar = build_covariance_from_scaling_rotation(scaling, scale_modifier, quat)  # (P, 3, 3)
    color = (SH0 * features_dc - color_rest_min).clamp(min=1e-20)       # (P, 1, 3)

    xyz_new = (gaussians.get_xyz + update.xyz).to(dtype)
    scaling_new = gaussians.scaling_activation((gaussians._scaling + update.scaling).to(dtype))
    quat_new = gaussians.rotation_activation((gaussians._rotation + update.rotation).to(dtype))
    features_dc_new = (features_dc + update.features_dc).to(dtype)
    features_rest_new = (features_rest + update.features_rest).to(dtype)
    opacity_new = gaussians.opacity_activation((gaussians._opacity + update.opacity).to(dtype))
    covar_new = build_covariance_from_scaling_rotation(scaling_new, scale_modifier, quat_new)
    color_new = (SH0 * features_dc_new - color_rest_min).clamp(min=1e-20)

    det_sqrt = scaling[:, 0:1] * scaling[:, 1:2] * scaling[:, 2:3]      # (P, 1)
    det_sqrt_new = scaling_new[:, 0:1] * scaling_new[:, 1:2] * scaling_new[:, 2:3]  # (P, 1)
    det_sqrt_avg = (0.5 * (det_sqrt ** 2 + det_sqrt_new ** 2)).sqrt()

    if scale_invariant:
        C = (opacity * color.sum(dim=-1)).clamp(min=1e-20)       # (P, 1)
        C_new = (opacity_new * color_new.sum(dim=-1)).clamp(min=1e-20)
    else:
        C = (opacity * color.sum(dim=-1) * det_sqrt).clamp(min=1e-20)       # (P, 1)
        C_new = (opacity_new * color_new.sum(dim=-1) * det_sqrt_new).clamp(min=1e-20)   # (P, 1)

    H_squared_scaling = C * (1.0 - \
                        det_sqrt.sqrt() * det_sqrt_new.sqrt() / (det_sqrt_avg).clamp(min=0.0))

    print(f"H squared scaling: max = {H_squared_scaling.max().item():.6f}, mean = {H_squared_scaling.mean().item():.6f}")

    if debug:
        safe_interact(local={**globals(), **locals()}, banner="hellinger_scaling_debug")

    return

def debug_hellinger(gaussians, update, debug=False):
    for param_group in ["xyz", "scaling", "rotation", "opacity", "features_dc", "features_rest"]:
        update_group = GaussianModelVector.zeros_like(gaussians)
        setattr(update_group, param_group, getattr(update, param_group).clone())

        hellinger_dist = compute_hellinger_distance(gaussians, update_group, debug=debug)
        print(f"H dist ({param_group}): max = {hellinger_dist.max().item():.6e}, mean = {hellinger_dist.mean().item():.6e}")

    total_hellinger_dist = compute_hellinger_distance(gaussians, update, debug=debug)
    print(f"H dist (total): max = {total_hellinger_dist.max().item():.6e}, mean = {total_hellinger_dist.mean().item():.6e}")


def compute_hellinger_distance(gaussians, update, scale_invariant=True, debug=False):
    """
    For two unnormalized gaussians denoted by P(x) = C * N(x; mu, Sigma) and Q(x) = C' * N(x; mu', Sigma'),
    the squared Hellinger distance is given by:
        H^2(P, Q) = 1/2 (C + C') - (C * C')^(1/2) * |Sigma|^(1/4) * |Sigma'|^(1/4) / |(Sigma + Sigma')/2|^(1/2) * exp(-1/8 (mu - mu')^T ((Sigma + Sigma')/2)^(-1) (mu - mu'))
    We want to clip updates such that H^2(P, Q) < threshold.
    Each update parameter introduces some contribution to H^2, and we will split the contributions
    between position, rotation, scaling, opacity, and color evenly.
    When computing the contributions, we will approximate the other parameters as constant.

    C is the mass of the gaussian splat, which is given by:
    opacity * color * det(Sigma)^(1/2)
    """

    dtype = torch.double
    SH0 = 0.282
    SH_rest = 1.0
    color_rest_min = -0.5
    scale_modifier = 1.0

    xyz = gaussians.get_xyz.to(dtype)                                             # (P, 3)
    scaling = gaussians.scaling_activation(gaussians._scaling.to(dtype))          # (P, 3)
    quat = gaussians.rotation_activation(gaussians._rotation.to(dtype))           # (P, 4)
    features_dc = gaussians.get_features_dc.to(dtype)                             # (P, 1, 3)
    features_rest = gaussians.get_features_rest.to(dtype)                         # (P, 15, 3)
    opacity = gaussians.opacity_activation(gaussians._opacity.to(dtype))          # (P, 1)
    covar = build_covariance_from_scaling_rotation(scaling, scale_modifier, quat)  # (P, 3, 3)
    color = (SH0 * features_dc - color_rest_min).clamp(min=1e-20)       # (P, 1, 3)

    xyz_new = (gaussians.get_xyz + update.xyz).to(dtype)
    scaling_new = gaussians.scaling_activation((gaussians._scaling + update.scaling).to(dtype))
    quat_new = gaussians.rotation_activation((gaussians._rotation + update.rotation).to(dtype))
    features_dc_new = (features_dc + update.features_dc).to(dtype)
    features_rest_new = (features_rest + update.features_rest).to(dtype)
    opacity_new = gaussians.opacity_activation((gaussians._opacity + update.opacity).to(dtype))
    covar_new = build_covariance_from_scaling_rotation(scaling_new, scale_modifier, quat_new)
    color_new = (SH0 * features_dc_new - color_rest_min).clamp(min=1e-20)

    det_sqrt = scaling[:, 0:1] * scaling[:, 1:2] * scaling[:, 2:3]      # (P, 1)
    det_sqrt_new = scaling_new[:, 0:1] * scaling_new[:, 1:2] * scaling_new[:, 2:3]  # (P, 1)
    covar_avg = 0.5 * (covar + covar_new)
    det_sqrt = torch.det(covar).clamp(min=1e-20).sqrt().unsqueeze(-1)         # (P, 1)
    det_sqrt_new = torch.det(covar_new).clamp(min=1e-20).sqrt().unsqueeze(-1) # (P, 1)
    det_sqrt_avg = torch.det(covar_avg).clamp(min=1e-20).sqrt().unsqueeze(-1) # (P, 1)

    if scale_invariant:
        C = opacity.clamp(min=1e-20)
        C_new = opacity_new.clamp(min=1e-20)
        # C = (opacity * color.sum(dim=-1)).clamp(min=1e-20)       # (P, 1)
        # C_new = (opacity_new * color_new.sum(dim=-1)).clamp(min=1e-20)
    else:
        pass
        # C = (opacity * color.sum(dim=-1) * det_sqrt).clamp(min=1e-20)       # (P, 1)
        # C_new = (opacity_new * color_new.sum(dim=-1) * det_sqrt_new).clamp(min=1e-20)   # (P, 1)

    covar_avg_reg = covar_avg + 1e-5 * torch.eye(3, device=covar_avg.device).unsqueeze(0)  # (P, 3, 3)

    xyz_delta = (xyz - xyz_new)
    mahalanobis = torch.linalg.solve(covar_avg_reg, xyz_delta)               # (P, 3)
    mahalanobis /= (scale_modifier ** 2)
    mahalanobis = torch.sum(mahalanobis * xyz_delta, dim=-1, keepdim=True)  # (P, 1)

    H_squared = 0.5 * (C + C_new) - torch.sqrt(C * C_new) * \
                det_sqrt.sqrt() * det_sqrt_new.sqrt() / \
                (det_sqrt_avg).clamp(min=1e-20) * \
                torch.exp(-0.125 * mahalanobis)

    if debug:
        safe_interact(local={**globals(), **locals()}, banner="hellinger_debug")

    H_squared.clip_(min=0.0)

    return H_squared

def compute_quat_to_trace_coefficient(quat_tilde, S):

    """
    Estimate the quadratic coeffient for the trace term in terms of the unnormalized quaternion.
    Here, the rotation needs to be transposed because that is how 3DGS handles it
    """

    p = quat_tilde.shape[0]

    w = quat_tilde[:,0]
    x = quat_tilde[:,1]
    y = quat_tilde[:,2]
    z = quat_tilde[:,3]
    q2 = (x**2 + y**2 + z**2 + w**2)
    q4 = q2 ** 2
    q6 = q4 * q2

    w = w[:, None, None]
    x = x[:, None, None]
    y = y[:, None, None]
    z = z[:, None, None]
    q2 = q2[:, None, None]
    q4 = q4[:, None, None]
    q6 = q6[:, None, None]

    R_tilde = quat_to_rot(quat_tilde, normalize=False).transpose(1, 2)
    R = R_tilde / q2

    coeffs = torch.zeros(p, 4, device=quat_tilde.device)

    for j in range(4):
        param = ["w", "x", "y", "z"][j]
        t = [w, x, y, z][j]

        dR_tilde = quat_to_drot(quat_tilde, wrt=param, normalize=False).transpose(1, 2)
        d2R_tilde = quat_to_d2rot(quat_tilde, wrt=param, normalize=False).transpose(1, 2)

        dG = R.transpose(1, 2) @ (dR_tilde / q2 - 2 * t * R_tilde / q4)
        d2G = R.transpose(1, 2) @ (-2 * t * dR_tilde / q4 + d2R_tilde / q2 + 8 * (t ** 2) * R_tilde / q6 - 2 * t * dR_tilde / q4 - 2 * R_tilde / q4)

        SdGSinv = S.unsqueeze(-1) * dG * (S ** -1).unsqueeze(1)
        coeff = 2 * ((SdGSinv * SdGSinv).sum(dim=(1,2)) + torch.diagonal(d2G, dim1=1, dim2=2).sum(dim=1))
        coeffs[:, j] = coeff

        # if j == 0:
        #     print("param =", param)
        #     print(w[0].item(), x[0].item(), y[0].item(), z[0].item())
        #     print((x**2 + y**2 + z**2 + w**2)[0].item())
        #     print(f"q4 = {q4[0].item()}, q_6 = {q6[0].item()}")
        #     print(f"R = \n{R[0]}\nR_tilde = \n{R_tilde[0]}")
        #     print(f"dR_tilde =\n{dR_tilde[0]}\nd2R_tilde = \nd{d2R_tilde[0]}\ndG = \n{dG[0]}\nSdGSinv = \n{SdGSinv[0]}\nd2G = \n{d2G[0]}\ncoeff = {coeff[0].item()} for param {param}")
        #     temp1 = dR_tilde / q2
        #     print(f"(dR_tilde / q2) = \n{temp1[0]}")
        #     temp1 = t * R_tilde / q4
        #     print(f"(t * R_tilde / q4) = \n{temp1[0]}")
        #     print("\n\n")

        #     safe_interact(local=locals(), banner="quat_to_trace_coefficient_debug")

    # safe_interact(local=locals(), banner="quat_to_trace_coefficient_debug")

    coeffs.clip_(min=1e-20)

    return coeffs

def quat_to_rot(quat, normalize=True):
    R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    q = (x**2 + y**2 + z**2 + w**2).sqrt()
    R[:, 0, 0] = q ** 2 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = q ** 2 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = q ** 2 - 2 * (x ** 2 + y ** 2)

    if normalize:
        R /= (q ** 2).unsqueeze(-1).unsqueeze(-1)

    return R

def quat_to_drot(quat, wrt="x", normalize=True):
    dR = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    q = (x**2 + y**2 + z**2 + w**2).sqrt()
    if wrt == "x":
        dR[:, 0, 0] = 2 * x
        dR[:, 0, 1] = 2 * y
        dR[:, 0, 2] = 2 * z
        dR[:, 1, 0] = 2 * y
        dR[:, 1, 1] = -2 * x
        dR[:, 1, 2] = -2 * w
        dR[:, 2, 0] = 2 * z
        dR[:, 2, 1] = 2 * w
        dR[:, 2, 2] = -2 * x
    if wrt == "y":
        dR[:, 0, 0] = -2 * y
        dR[:, 0, 1] = 2 * x
        dR[:, 0, 2] = 2 * w
        dR[:, 1, 0] = 2 * x
        dR[:, 1, 1] = 2 * y
        dR[:, 1, 2] = 2 * z
        dR[:, 2, 0] = -2 * w
        dR[:, 2, 1] = 2 * z
        dR[:, 2, 2] = -2 * y
    if wrt == "z":
        dR[:, 0, 0] = -2 * z
        dR[:, 0, 1] = -2 * w
        dR[:, 0, 2] = 2 * x
        dR[:, 1, 0] = 2 * w
        dR[:, 1, 1] = -2 * z
        dR[:, 1, 2] = 2 * y
        dR[:, 2, 0] = 2 * x
        dR[:, 2, 1] = 2 * y
        dR[:, 2, 2] = 2 * z
    if wrt == "w":
        dR[:, 0, 0] = 2 * w
        dR[:, 0, 1] = -2 * z
        dR[:, 0, 2] = 2 * y
        dR[:, 1, 0] = 2 * z
        dR[:, 1, 1] = 2 * w
        dR[:, 1, 2] = -2 * x
        dR[:, 2, 0] = -2 * y
        dR[:, 2, 1] = 2 * x
        dR[:, 2, 2] = 2 * w
    if normalize:
        dR /= q.unsqueeze(-1).unsqueeze(-1)
    return dR

def quat_to_d2rot(quat, wrt="x", normalize=True):
    d2R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    q = (x**2 + y**2 + z**2 + w**2).sqrt()
    if wrt == "x":
        d2R[:, 0, 0] = 2
        d2R[:, 1, 1] = -2
        d2R[:, 2, 2] = -2
    if wrt == "y":
        d2R[:, 0, 0] = -2
        d2R[:, 1, 1] = 2
        d2R[:, 2, 2] = -2
    if wrt == "z":
        d2R[:, 0, 0] = -2
        d2R[:, 1, 1] = -2
        d2R[:, 2, 2] = 2
    if wrt == "w":
        d2R[:, 0, 0] = 2
        d2R[:, 1, 1] = 2
        d2R[:, 2, 2] = 2
    return d2R
