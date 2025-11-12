import math
import torch
from torch.utils.checkpoint import checkpoint
import time

from gaussian_renderer.render_hessian import render_hessian
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel

def scalar_training_loss_hessian(iteration, opt, viewpoint_cam, gaussians, pipe, bg, train_test_exp,
                         depth_l1_weight, batch_stats=None,
                         SPARSE_ADAM_AVAILABLE=False, FUSED_SSIM_AVAILABLE=False, 
                         pixel_mask=None
                         ):

    render_pkg = render_hessian(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    if batch_stats is not None:
        batch_stats["visibility_mask"][visibility_filter] = True
        batch_stats["max_radii2D"][visibility_filter] = torch.max(batch_stats["max_radii2D"][visibility_filter], radii[visibility_filter])

    if viewpoint_cam.alpha_mask is not None:
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        image *= alpha_mask

    # Loss
    gt_image = viewpoint_cam.original_image.cuda()

    # Apply pixel sampling. Set unselected pixels to gt_image to avoid affecting loss
    if pixel_mask is not None:
        image[:,pixel_mask] = gt_image[:,pixel_mask]

    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)

    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

    # Depth regularization
    Ll1depth_pure = 0.0
    if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        invDepth = render_pkg["depth"]
        mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        depth_mask = viewpoint_cam.depth_mask.cuda()

        Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
        loss += Ll1depth
        Ll1depth = Ll1depth.item()
    else:
        Ll1depth = 0

    return loss, Ll1, Ll1depth
