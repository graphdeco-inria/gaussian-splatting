import math
import torch
import torch.autograd.forward_ad as fwAD
from functools import partial

from solver.gaussian_model_vector import GaussianModelVector
from solver.training_loss import scalar_training_loss
from solver.batch_training_loss import batch_training_loss
from solver.training_loss_hessian import scalar_training_loss_hessian
from utils.general_utils import safe_interact

from solver.gaussian_model_vector import GaussianModelVector

def Dhat(gaussians, viewpoint_cams, scale=1.0, **render_kwargs):
    squared_weights = None
    with torch.no_grad():
        for vc in viewpoint_cams:
            _, batch_stats = batch_training_loss(gaussians=gaussians, viewpoint_cams=[vc], return_stats=True, track_weights=True, **render_kwargs)

            if squared_weights is None:
                squared_weights = batch_stats["squared_weights"]
            else:
                for k in squared_weights.keys():
                    squared_weights[k] += batch_stats["squared_weights"][k]

        D = GaussianModelVector(xyz=squared_weights["means3D"],
                                features_dc=squared_weights["sh"][:,0:1,:],
                                features_rest=squared_weights["sh"][:,1:,:],
                                scaling=squared_weights["scales"],
                                rotation=squared_weights["rotations"],
                                opacity=squared_weights["opacities"],
                                exposure=0.0 * gaussians.get_exposure,
                                gaussians=gaussians)
    return D * scale

def loss(gaussians, viewpoint_cams, scale=1.0, **render_kwargs):
    with torch.no_grad():
        l = 0.0
        for vc in viewpoint_cams:
            # l += scalar_training_loss(gaussians=gaussians, viewpoint_cam=vc, **render_kwargs) ** 2
            l += batch_training_loss(gaussians=gaussians, viewpoint_cams=[vc], **render_kwargs).loss_scalar
        return l * scale

def g(gaussians, viewpoint_cams, scale=1.0, return_loss=False, **render_kwargs):
    gaussians.zero_grad()

    with torch.enable_grad():
        l = 0.0
        for vc in viewpoint_cams:
            # li = scalar_training_loss(gaussians=gaussians, viewpoint_cam=vc, **render_kwargs) ** 2
            li = batch_training_loss(gaussians=gaussians, viewpoint_cams=[vc], **render_kwargs).loss_scalar
            li *= scale

            l += li.item()
            li.backward(retain_graph=False)

    grad = GaussianModelVector.from_gaussians_grad(gaussians)

    if return_loss:
        return l, grad

    return grad

def JTJv(v, gaussians, viewpoint_cams, scale=1.0, S=None, damp=None, **render_kwargs):
    """
    Computes (scale * STJTJS + damp*I) v.
    """

    if S is not None:
        Sv = S * v
    else:
        Sv = v


    B = len(viewpoint_cams)
    batch_size = render_kwargs.get("batch_size", 1)
    batch_size = B if batch_size < 0 else batch_size

    gaussians.zero_grad()

    for start_idx in range(0, B, batch_size):
        with torch.enable_grad(), fwAD.dual_level(), gaussians.make_dual(Sv):
            end_idx = min(start_idx + batch_size, B)
            viewpoint_cams_batch = [viewpoint_cams[i] for i in range(start_idx, end_idx)]
            loss_dual = batch_training_loss(gaussians=gaussians, viewpoint_cams=viewpoint_cams_batch, **render_kwargs)
            loss_primal, loss_tangent = loss_dual.unpack_dual()
            loss_primal.backward(loss_tangent, retain_graph=False)

    JTJv = GaussianModelVector.from_gaussians_grad(gaussians) * scale

    if S is not None:
        JTJv = S * JTJv

    if damp is not None:
        JTJv += damp * v

    return JTJv

def dot(v1, v2):
    return v1.dot(v2)

def saxpy(a, x, y):
    return a * x + y

def construct_Dhat_func(**render_kwargs):
    return partial(Dhat, **render_kwargs)

def construct_loss_func(**render_kwargs):
    return partial(loss, **render_kwargs)

def construct_g_func(**render_kwargs):
    return partial(g, **render_kwargs)

def construct_Jv_func(**render_kwargs):
    return partial(Jv, **render_kwargs)

def construct_JTJv_func(**render_kwargs):
    return partial(JTJv, **render_kwargs)
