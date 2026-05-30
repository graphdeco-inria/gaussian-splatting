import math
import torch
from torch.utils.checkpoint import checkpoint
import time

from solver.loss_image_state import BatchLossImageState
from gaussian_renderer.batch_render import batch_render
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel
from utils.general_utils import safe_interact

try:
    from fused_ssim import fused_ssim_per_pixel, fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def huber_loss(x, delta=0.1):
    x_abs = x.abs()
    mask = x_abs <= delta
    x_abs[~mask] = (2 * delta * x_abs[~mask] - delta ** 2).sqrt()
    return x_abs

def compute_batch_loss_block(images, alpha_masks, gt_images, per_image_alphas, per_image_betas, FUSED_SSIM_AVAILABLE=False, **kwargs): # use_l1_loss=False, disable_ssim=False):
    disable_ssim = kwargs.get("disable_ssim", False)
    loss_type = kwargs.get("loss_type", "l2")
    huber_delta = kwargs.get("huber_delta", 0.1)
    regularize = kwargs.get("regularize", False)

    if alpha_masks is not None:
        images = images * alpha_masks

    Ll1_per_pixel = images - gt_images

    if disable_ssim:
        ssim_loss_per_pixel = torch.zeros_like(Ll1_per_pixel)
    else:
        if FUSED_SSIM_AVAILABLE:
            # raise NotImplementedError("Fused SSIM is not implemented in this version.")
            ssim_value = fused_ssim_per_pixel(images, gt_images)
        else:
            ssim_value = ssim_per_pixel(images, gt_images)

        # ssim_loss_per_pixel = 1.0 - ssim_per_pixel(images, gt_images)
        ssim_loss_per_pixel = 1.0 - ssim_value

    if loss_type == "l2":
        pass
    elif loss_type == "l1":
        Ll1_per_pixel = Ll1_per_pixel.abs()
        Ll1_mask = Ll1_per_pixel != 0.0
        Ll1_per_pixel[Ll1_mask] = torch.sqrt(Ll1_per_pixel[Ll1_mask])

        ssim_loss_per_pixel = ssim_loss_per_pixel.abs()
        ssim_mask = ssim_loss_per_pixel != 0.0
        ssim_loss_per_pixel[ssim_mask] = torch.sqrt(ssim_loss_per_pixel[ssim_mask])
    elif loss_type == "huber":
        Ll1_per_pixel = huber_loss(Ll1_per_pixel, delta=huber_delta)
        ssim_loss_per_pixel = huber_loss(ssim_loss_per_pixel, delta=huber_delta)


    Ll1_per_pixel = per_image_alphas * Ll1_per_pixel
    ssim_loss_per_pixel = per_image_betas * ssim_loss_per_pixel

    return Ll1_per_pixel, ssim_loss_per_pixel

def batch_training_loss(iteration, opt, viewpoint_cams, gaussians, pipe, bg, train_test_exp,
                        depth_l1_weight, return_stats=False,
                        SPARSE_ADAM_AVAILABLE=False, FUSED_SSIM_AVAILABLE=False, 
                        pixel_mask=None,
                        track_weights=False,
                        **kwargs
                        ):
    scale_reg = opt.scale_reg
    opacity_reg = opt.opacity_reg
    color_reg = opt.color_reg

    B = len(viewpoint_cams)

    sizes_list = [(vc.image_height, vc.image_width) for vc in viewpoint_cams]

    max_H = max(s[0] for s in sizes_list)
    max_W = max(s[1] for s in sizes_list)

    batch_render_pkg = batch_render(viewpoint_cams, gaussians, pipe, bg, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, track_weights=track_weights)

    images, viewspace_point_tensor, visibility_filter, max_radii, squared_weights = batch_render_pkg["render"], batch_render_pkg["viewspace_points"], batch_render_pkg["visibility_filter"], batch_render_pkg["max_radii"], batch_render_pkg["squared_weights"]

    if return_stats:
        batch_stats = {}
        batch_stats['viewspace_point_tensor'] = viewspace_point_tensor
        batch_stats['visibility_filter'] = visibility_filter
        batch_stats['max_radii'] = max_radii
        batch_stats['viewcount'] = batch_render_pkg.get('viewcount', None)
        batch_stats['squared_weights'] = squared_weights
        batch_stats['images'] = images

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
        alpha_per_image.append(math.sqrt(2.0 * alpha / n))
        beta_per_image.append(math.sqrt(2.0 * beta / n))
    alpha_per_image = torch.tensor(alpha_per_image, dtype=images.dtype, device=images.device).view(B, 1, 1, 1)
    beta_per_image = torch.tensor(beta_per_image, dtype=images.dtype, device=images.device).view(B, 1, 1, 1)
    
    # TODO: get checkpointing to work here
    Ll1_per_pixel, ssim_loss_per_pixel = compute_batch_loss_block(images, alpha_masks, gt_images, alpha_per_image, beta_per_image, FUSED_SSIM_AVAILABLE, **kwargs) # use_l1_loss, disable_ssim)

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
        pass
        # loss_image = torch.cat((Ll1_per_pixel, ssim_loss_per_pixel), dim=1)


    if scale_reg > 0.0:
        scaling = gaussians.get_scaling
        scaling_numel = scaling.numel()
        scaling_nonzero = scaling[scaling > 0.0]
        scaling_reg_loss = math.sqrt(2.0 * scale_reg / scaling_numel) * scaling_nonzero.sqrt()
    else:
        scaling_reg_loss = torch.tensor([], device=images.device)

    if opacity_reg > 0.0:
        opacity = gaussians.get_opacity
        opacity_numel = opacity.numel()
        opacity_nonzero = opacity[opacity > 0.0]
        if opt.binarize_opacity_reg:
            opacity_reg_loss = math.sqrt(2.0 * opacity_reg / opacity_numel) * (opacity_nonzero * (1 - opacity_nonzero)).sqrt()
        else:
            opacity_reg_loss = math.sqrt(2.0 * opacity_reg / opacity_numel) * opacity_nonzero.sqrt()
    else:
        opacity_reg_loss = torch.tensor([], device=images.device)

    if color_reg > 0.0:
        color_dc = gaussians.get_color_dc.clamp(min=1e-5)
        color_numel = color_dc.numel()
        color_nonzero = color_dc[color_dc > 0.0]
        color_reg_loss = math.sqrt(2.0 * color_reg / color_numel) * color_nonzero.sqrt()
    else:
        color_reg_loss = torch.tensor([], device=images.device)


    loss_image = torch.cat((Ll1_per_pixel.flatten(), ssim_loss_per_pixel.flatten(), scaling_reg_loss.flatten(), opacity_reg_loss.flatten(), color_reg_loss.flatten()), dim=0)

    if kwargs.get("debug_loss", False):
        safe_interact(local=locals(), banner="Debug batch_training_loss")

    if return_stats:
        return loss_image, batch_stats
    
    return loss_image

