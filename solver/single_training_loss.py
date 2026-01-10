import math
import torch
from torch.utils.checkpoint import checkpoint
import torch.autograd.forward_ad as fwAD
import time

from solver.loss_image_state import BatchLossImageState
from gaussian_renderer.single_render import single_render as batch_render
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel
from utils.general_utils import safe_interact

def has_tangent(x):
    if isinstance(x, float):
        return False
    elif isinstance(x, list):
        return any(has_tangent(xi) for xi in x)
    return fwAD.unpack_dual(x).tangent is not None

def get_tangent(x):
    if isinstance(x, float):
        return 0.0
    elif isinstance(x, list):
        return [get_tangent(xi) for xi in x]
    elif has_tangent(x):
        return fwAD.unpack_dual(x).tangent
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    else:
        raise ValueError(f"Unsupported type for tangent extraction: {type(x)}")

def get_primal(x):
    if has_tangent(x):
        return fwAD.unpack_dual(x).primal
    else:
        return x

def apply_sqrt(input):
    jvp = has_tangent(input)
    return Sqrt.apply(input, jvp)

class Sqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, jvp):
        ctx.save_for_backward(input)
        if jvp:
            ctx.save_for_forward(get_primal(input))
        return input.sqrt()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output / (2 * input.sqrt()).clamp_min(torch.finfo(grad_output.dtype).eps)
        return grad_input, None

    @staticmethod
    def jvp(ctx, grad_input, grad_jvp):
        input, = ctx.saved_tensors
        return grad_input / (2 * input.sqrt()).clamp_min(torch.finfo(grad_input.dtype).eps)

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
            ssim_value = fused_ssim(images, gt_images)
        else:
            ssim_value = ssim_per_pixel(images, gt_images)

        ssim_loss_per_pixel = 1.0 - ssim_per_pixel(images, gt_images)

    if loss_type == "l2":
        pass
    elif loss_type == "l1":
        Ll1_per_pixel = apply_sqrt(Ll1_per_pixel.abs())
        ssim_loss_per_pixel = apply_sqrt(ssim_loss_per_pixel.abs())

        # Ll1_per_pixel = Ll1_per_pixel.abs()
        # Ll1_mask = Ll1_per_pixel != 0.0
        # Ll1_per_pixel[Ll1_mask] = torch.sqrt(Ll1_per_pixel[Ll1_mask])

        # ssim_loss_per_pixel = ssim_loss_per_pixel.abs()
        # ssim_mask = ssim_loss_per_pixel != 0.0
        # ssim_loss_per_pixel[ssim_mask] = torch.sqrt(ssim_loss_per_pixel[ssim_mask])
    elif loss_type == "huber":
        Ll1_per_pixel = huber_loss(Ll1_per_pixel, delta=huber_delta)
        ssim_loss_per_pixel = huber_loss(ssim_loss_per_pixel, delta=huber_delta)

    Ll1_per_pixel = per_image_alphas * Ll1_per_pixel
    ssim_loss_per_pixel = per_image_betas * ssim_loss_per_pixel

    return Ll1_per_pixel, ssim_loss_per_pixel

def single_training_loss(iteration, opt, viewpoint_cams, gaussians, pipe, bg, train_test_exp,
                        depth_l1_weight, return_stats=False,
                        SPARSE_ADAM_AVAILABLE=False, FUSED_SSIM_AVAILABLE=False, 
                        pixel_mask=None,
                        track_weights=False,
                        **kwargs
                        ):
    scale_reg = kwargs.get('scale_reg', 0.0)
    opacity_reg = kwargs.get('opacity_reg', 0.0)

    B = len(viewpoint_cams)

    assert B == 1, "batch_training_loss currently only supports batch size of 1."

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

    gt_images = viewpoint_cams[0].original_image.cuda().unsqueeze(0)
    alpha_masks = None
    if viewpoint_cams[0].alpha_mask is not None:
        alpha_masks = viewpoint_cams[0].alpha_mask.cuda()


    alpha, beta = 1.0 - opt.lambda_dssim, opt.lambda_dssim
    n = 3 * max_H * max_W

    alpha_per_image = math.sqrt(2.0 * alpha / n)
    beta_per_image = math.sqrt(2.0 * beta / n)
    
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
        scaling_reg_loss = math.sqrt(2.0 * scale_reg / scaling.numel()) * apply_sqrt(scaling)
        # scaling_numel = scaling.numel()
        # scaling_nonzero = scaling[scaling > 0.0]
        # scaling_reg_loss = math.sqrt(scale_reg / scaling_numel) * scaling_nonzero.sqrt()
    else:
        scaling_reg_loss = torch.tensor([], device=images.device)

    if opacity_reg > 0.0:
        opacity = gaussians.get_opacity
        opacity_reg_loss = math.sqrt(2.0 * opacity_reg / opacity.numel()) * apply_sqrt(opacity)
        # opacity_numel = opacity.numel()
        # opacity_nonzero = opacity[opacity > 0.0]
        # opacity_reg_loss = math.sqrt(opacity_reg / opacity_numel) * opacity_nonzero.sqrt()
    else:
        opacity_reg_loss = torch.tensor([], device=images.device)

    loss_image = torch.cat((Ll1_per_pixel.flatten(), ssim_loss_per_pixel.flatten(), scaling_reg_loss.flatten(), opacity_reg_loss.flatten()), dim=0)

    if kwargs.get("debug_loss", False):
        safe_interact(local=locals(), banner="Debug batch_training_loss")

    if return_stats:
        return loss_image, batch_stats
    
    return loss_image

