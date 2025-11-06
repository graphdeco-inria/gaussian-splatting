import math
import torch
from torch.utils.checkpoint import checkpoint
import time

from solver.loss_image_state import BatchLossImageState
from gaussian_renderer.batch_render import batch_render
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel

def compute_batch_loss_block(images, alpha_masks, gt_images, per_image_alphas, per_image_betas, FUSED_SSIM_AVAILABLE=False, disable_ssim=False):
    if alpha_masks is not None:
        images = images * alpha_masks


    if disable_ssim:
        Ll1_per_pixel = (images - gt_images)
        ssim_loss_per_pixel = Ll1_per_pixel
        Ll1_per_pixel = per_image_alphas * Ll1_per_pixel
        ssim_loss_per_pixel = per_image_betas * ssim_loss_per_pixel

    else:
        Ll1_per_pixel = l1_loss_per_pixel(images, gt_images)
        if FUSED_SSIM_AVAILABLE:
            # raise NotImplementedError("Fused SSIM is not implemented in this version.")
            ssim_value = fused_ssim(images, gt_images)
        else:
            ssim_value = ssim_per_pixel(images, gt_images)

        ssim_loss_per_pixel = 1.0 - ssim_per_pixel(images, gt_images)
        ssim_loss_per_pixel = ssim_loss_per_pixel.abs()     # This is not in the original implementation, but it should be there to avoid NaNs

        Ll1_mask = Ll1_per_pixel != 0.0
        Ll1_per_pixel[Ll1_mask] = torch.sqrt(Ll1_per_pixel[Ll1_mask])
        Ll1_per_pixel = per_image_alphas * Ll1_per_pixel

        ssim_mask = ssim_loss_per_pixel != 0.0
        ssim_loss_per_pixel[ssim_mask] = torch.sqrt(ssim_loss_per_pixel[ssim_mask])
        ssim_loss_per_pixel = per_image_betas * ssim_loss_per_pixel

        # Ll1_per_pixel = per_image_alphas * torch.sqrt(Ll1_per_pixel)
        # ssim_loss_per_pixel = per_image_betas * torch.sqrt(ssim_loss_per_pixel)
    return Ll1_per_pixel, ssim_loss_per_pixel

def batch_training_loss(iteration, opt, viewpoint_cams, gaussians, pipe, bg, train_test_exp,
                        depth_l1_weight, batch_stats=None,
                        SPARSE_ADAM_AVAILABLE=False, FUSED_SSIM_AVAILABLE=False, 
                        disable_ssim=False, 
                        pixel_mask=None
                        ):

    B = len(viewpoint_cams)

    sizes_list = [(vc.image_height, vc.image_width) for vc in viewpoint_cams]

    max_H = max(s[0] for s in sizes_list)
    max_W = max(s[1] for s in sizes_list)

    batch_render_pkg = batch_render(viewpoint_cams, gaussians, pipe, bg, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

    images, viewspace_point_tensor, visibility_filter, max_radii = batch_render_pkg["render"], batch_render_pkg["viewspace_points"], batch_render_pkg["visibility_filter"], batch_render_pkg["max_radii"]

    if batch_stats is not None:
        batch_stats['viewspace_point_tensor'] = viewspace_point_tensor
        batch_stats['visibility_filter'] = visibility_filter
        batch_stats['max_radii'] = max_radii
        batch_stats['viewcount'] = batch_render_pkg.get('viewcount', None)

    gt_images = torch.zeros_like(images)
    for i, vc in enumerate(viewpoint_cams):
        H, W = vc.image_height, vc.image_width
        gt_images[i, :, :H, :W] = vc.original_image.cuda()

        # Apply pixel sampling. Set unselected pixels to gt_image to avoid affecting loss
        if pixel_mask is not None:
            images[i,:,pixel_mask[i]] = gt_images[i,:,pixel_mask[i]]

    alpha_masks = None
    if any([vc.alpha_mask is not None for vc in viewpoint_cams]):
        alpha_masks = torch.zeros((B, 1, max_H, max_W), device="cuda")

        for i, vc in enumerate(viewpoint_cams):
            if vc.alpha_mask is not None:
                alpha_masks[i] = vc.alpha_mask.cuda()

    alpha, beta = 1.0 - opt.lambda_dssim, opt.lambda_dssim
    alpha_per_image, beta_per_image = [], []
    for vc in viewpoint_cams:
        H, W = int(vc.image_height), int(vc.image_width)
        n = 3 * H * W
        alpha_per_image.append(math.sqrt(alpha / n))
        beta_per_image.append(math.sqrt(beta / n))
    alpha_per_image = torch.tensor(alpha_per_image, dtype=images.dtype, device=images.device).view(B, 1, 1, 1)
    beta_per_image = torch.tensor(beta_per_image, dtype=images.dtype, device=images.device).view(B, 1, 1, 1)

    
    # TODO: get checkpointing to work here
    Ll1_per_pixel, ssim_loss_per_pixel = compute_batch_loss_block(images, alpha_masks, gt_images, alpha_per_image, beta_per_image, FUSED_SSIM_AVAILABLE, disable_ssim)

    has_depth = any([vc.depth_reliable for vc in viewpoint_cams])

    # Depth regularization
    Ll1depth_pure = 0.0
    if depth_l1_weight(iteration) > 0 and has_depth:
        invDepth = render_pkg["depth"]
        mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        depth_mask = viewpoint_cam.depth_mask.cuda()

        Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
        loss += Ll1depth
        Ll1depth = Ll1depth.item()

        raise NotImplementedError("Ll1depth_pure_per_pixel is not implemented in this version.")

    else:
        loss_image = torch.cat((Ll1_per_pixel, ssim_loss_per_pixel), dim=1)

    loss_image_state = BatchLossImageState(loss_image, sizes_list, has_depth)

    return loss_image_state

