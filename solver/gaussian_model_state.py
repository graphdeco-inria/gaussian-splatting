import torch
import torch.autograd.forward_ad as fwAD

class GaussianModelScaleMatrix:
    """
    Per parameter damping matrix for Gaussians
    """
    def __init__(self, xyz_scale, features_dc_scale, features_rest_scale,
                 scaling_scale, rotation_scale, opacity_scale, exposure_scale):
        self.xyz_scale = xyz_scale
        self.features_dc_scale = features_dc_scale
        self.features_rest_scale = features_rest_scale
        self.scaling_scale = scaling_scale
        self.rotation_scale = rotation_scale
        self.opacity_scale = opacity_scale
        self.exposure_scale = exposure_scale

    def __neg__(self):
        return GaussianModelScaleMatrix(-self.xyz_scale,
                                       -self.features_dc_scale,
                                       -self.features_rest_scale,
                                       -self.scaling_scale,
                                       -self.rotation_scale,
                                       -self.opacity_scale,
                                       -self.exposure_scale)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelScaleMatrix(self.xyz_scale + other,
                                           self.features_dc_scale + other,
                                           self.features_rest_scale + other,
                                           self.scaling_scale + other,
                                           self.rotation_scale + other,
                                           self.opacity_scale + other,
                                           self.exposure_scale + other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelScaleMatrix(self.xyz_scale * other,
                                           self.features_dc_scale * other,
                                           self.features_rest_scale * other,
                                           self.scaling_scale * other,
                                           self.rotation_scale * other,
                                           self.opacity_scale * other,
                                           self.exposure_scale * other)
        if isinstance(other, GaussianModelState):
            return other * self
        else:
            raise TypeError(f"Can only multiply by scalar values, not {type(other)}")

class GaussianModelParamGroupMask:
    """
    Per parameter mask
    """
    def __init__(self, mask_xyz=False, mask_features_dc=False, mask_features_rest=False,
                 mask_scaling=False, mask_rotation=False, mask_opacity=False, mask_exposure=False):
        self.mask_xyz = mask_xyz
        self.mask_features_dc = mask_features_dc
        self.mask_features_rest = mask_features_rest
        self.mask_scaling = mask_scaling
        self.mask_rotation = mask_rotation
        self.mask_opacity = mask_opacity
        self.mask_exposure = mask_exposure

class GaussianModelSplatMask:
    """
    Per splat mask
    """
    def __init__(self, mask_out_filter):
        self.mask_out_filter = mask_out_filter

class GaussianModelState:
    """
    Represents updates to Gaussian parameters as a generalized vector
    """

    def __init__(self, xyz_grad, features_dc_grad, features_rest_grad,
                 scaling_grad, rotation_grad, opacity_grad, exposure_grad,
                 param_mask=None, splat_mask=None):
        self.xyz_grad = xyz_grad
        self.features_dc_grad = features_dc_grad
        self.features_rest_grad = features_rest_grad
        self.scaling_grad = scaling_grad
        self.rotation_grad = rotation_grad
        self.opacity_grad = opacity_grad
        self.exposure_grad = exposure_grad

        if param_mask is not None:
            assert isinstance(param_mask, GaussianModelParamGroupMask), "param_mask must be an instance of GaussianModelParamGroupMask"
            if param_mask.mask_xyz:
                self.xyz_grad.zero_()
            if param_mask.mask_features_dc:
                self.features_dc_grad.zero_()
            if param_mask.mask_features_rest:
                self.features_rest_grad.zero_()
            if param_mask.mask_scaling:
                self.scaling_grad.zero_()
            if param_mask.mask_rotation:
                self.rotation_grad.zero_()
            if param_mask.mask_opacity:
                self.opacity_grad.zero_()
            if param_mask.mask_exposure:
                self.exposure_grad.zero_()
        if splat_mask is not None:
            self.xyz_grad[splat_mask.mask_out_filter] = 0.0
            self.features_dc_grad[splat_mask.mask_out_filter] = 0.0
            self.features_rest_grad[splat_mask.mask_out_filter] = 0.0
            self.scaling_grad[splat_mask.mask_out_filter] = 0.0
            self.rotation_grad[splat_mask.mask_out_filter] = 0.0
            self.opacity_grad[splat_mask.mask_out_filter] = 0.0

    @classmethod
    def zero_like_gaussians(cls, gaussians, param_mask=None, splat_mask=None):
        return cls(torch.zeros_like(gaussians._xyz),
                   torch.zeros_like(gaussians._features_dc),
                   torch.zeros_like(gaussians._features_rest),
                   torch.zeros_like(gaussians._scaling),
                   torch.zeros_like(gaussians._rotation),
                   torch.zeros_like(gaussians._opacity),
                   torch.zeros_like(gaussians._exposure),
                   param_mask=param_mask,
                   splat_mask=splat_mask)

    @classmethod
    def rademacher_like_gaussians(cls, gaussians, param_mask=None, splat_mask=None):
        def rademacher_like(T):
            return (2 * torch.randint(0, 2, T.shape, device=T.device, dtype=torch.int8) - 1).to(T.dtype)
        return cls(rademacher_like(gaussians._xyz),
                   rademacher_like(gaussians._features_dc),
                   rademacher_like(gaussians._features_rest),
                   rademacher_like(gaussians._scaling),
                   rademacher_like(gaussians._rotation),
                   rademacher_like(gaussians._opacity),
                   rademacher_like(gaussians._exposure),
                   param_mask=param_mask,
                   splat_mask=splat_mask)

    @classmethod
    def randn_like_gaussians(cls, gaussians, param_mask=None, splat_mask=None):
        def randn_like(T):
            return torch.randn(T.shape, device=T.device, dtype=T.dtype)
        return cls(randn_like(gaussians._xyz),
                   randn_like(gaussians._features_dc),
                   randn_like(gaussians._features_rest),
                   randn_like(gaussians._scaling),
                   randn_like(gaussians._rotation),
                   randn_like(gaussians._opacity),
                   randn_like(gaussians._exposure),
                   param_mask=param_mask,
                   splat_mask=splat_mask)

    @classmethod
    def from_gaussians(cls, gaussians, param_mask=None, splat_mask=None):
        return cls(gaussians._xyz.data.clone(),
                   gaussians._features_dc.data.clone(),
                   gaussians._features_rest.data.clone(),
                   gaussians._scaling.data.clone(),
                   gaussians._rotation.data.clone(),
                   gaussians._opacity.data.clone(),
                   gaussians._exposure.data.clone(),
                   param_mask=param_mask,
                   splat_mask=splat_mask)

    @classmethod
    def from_gaussians_grad(cls, gaussians, param_mask=None, splat_mask=None):
        xyz_grad = gaussians._xyz.grad if gaussians._xyz.grad is not None else torch.zeros_like(gaussians._xyz)
        features_dc_grad = gaussians._features_dc.grad if gaussians._features_dc.grad is not None else torch.zeros_like(gaussians._features_dc)
        features_rest_grad = gaussians._features_rest.grad if gaussians._features_rest.grad is not None else torch.zeros_like(gaussians._features_rest)
        scaling_grad = gaussians._scaling.grad if gaussians._scaling.grad is not None else torch.zeros_like(gaussians._scaling)
        rotation_grad = gaussians._rotation.grad if gaussians._rotation.grad is not None else torch.zeros_like(gaussians._rotation)
        opacity_grad = gaussians._opacity.grad if gaussians._opacity.grad is not None else torch.zeros_like(gaussians._opacity)
        exposure_grad = gaussians._exposure.grad if gaussians._exposure.grad is not None else torch.zeros_like(gaussians._exposure)
        return cls(xyz_grad,
                   features_dc_grad,
                   features_rest_grad,
                   scaling_grad,
                   rotation_grad,
                   opacity_grad,
                   exposure_grad,
                   param_mask=param_mask,
                   splat_mask=splat_mask)

    @classmethod
    def from_tangent_grad(cls, gaussians_model_state, param_mask=None, splat_mask=None):
        xyz_grad = gaussians_model_state.xyz_grad.grad if gaussians_model_state.xyz_grad.grad is not None else torch.zeros_like(gaussians_model_state.xyz_grad)
        features_dc_grad = gaussians_model_state.features_dc_grad.grad if gaussians_model_state.features_dc_grad.grad is not None else torch.zeros_like(gaussians_model_state.features_dc_grad)
        features_rest_grad = gaussians_model_state.features_rest_grad.grad if gaussians_model_state.features_rest_grad.grad is not None else torch.zeros_like(gaussians_model_state.features_rest_grad)
        scaling_grad = gaussians_model_state.scaling_grad.grad if gaussians_model_state.scaling_grad.grad is not None else torch.zeros_like(gaussians_model_state.scaling_grad)
        rotation_grad = gaussians_model_state.rotation_grad.grad if gaussians_model_state.rotation_grad.grad is not None else torch.zeros_like(gaussians_model_state.rotation_grad)
        opacity_grad = gaussians_model_state.opacity_grad.grad if gaussians_model_state.opacity_grad.grad is not None else torch.zeros_like(gaussians_model_state.opacity_grad)
        exposure_grad = gaussians_model_state.exposure_grad.grad if gaussians_model_state.exposure_grad.grad is not None else torch.zeros_like(gaussians_model_state.exposure_grad)
        return cls(xyz_grad,
                   features_dc_grad,
                   features_rest_grad,
                   scaling_grad,
                   rotation_grad,
                   opacity_grad,
                   exposure_grad,
                   param_mask=param_mask,
                   splat_mask=splat_mask)


    def requires_grad_(self, requires_grad=True):
        self.xyz_grad.requires_grad_(requires_grad)
        self.features_dc_grad.requires_grad_(requires_grad)
        self.features_rest_grad.requires_grad_(requires_grad)
        self.scaling_grad.requires_grad_(requires_grad)
        self.rotation_grad.requires_grad_(requires_grad)
        self.opacity_grad.requires_grad_(requires_grad)
        self.exposure_grad.requires_grad_(requires_grad)
        return self

    def apply_mask(self, param_mask=None, splat_mask=None):
        return GaussianModelState(
                self.xyz_grad,
                self.features_dc_grad,
                self.features_rest_grad,
                self.scaling_grad,
                self.rotation_grad,
                self.opacity_grad,
                self.exposure_grad,
                param_mask=param_mask,
                splat_mask=splat_mask)


    def clip(self):
        feat_dc_grad_norm = self.features_dc_grad.norm()
        feat_dc_grad_norm_max = self.features_dc_grad.norm(dim=-1).max()
        feat_dc_grad_max = self.features_dc_grad.abs().max()
        measured_norm = feat_dc_grad_norm
        target_norm = min(measured_norm, 8.0)
        print(f"Max per-gaussian features_dc grad norm: {feat_dc_grad_norm:.4e}")
        print(f"Max per-gaussian features_dc grad norm: {feat_dc_grad_norm_max:.4e}")
        print(f"Max per-gaussian features_dc grad abs: {feat_dc_grad_max:.4e}")
        print(f"Clipping ratio: {target_norm / measured_norm:.4e}")
        self.xyz_grad *= target_norm / measured_norm
        self.features_dc_grad *= target_norm / measured_norm
        self.features_rest_grad *= target_norm / measured_norm
        self.scaling_grad *= target_norm / measured_norm
        self.rotation_grad *= target_norm / measured_norm
        self.opacity_grad *= target_norm / measured_norm
        self.exposure_grad *= target_norm / measured_norm

    def block_average_and_expand(self):
        self.xyz_grad = self.xyz_grad.mean(dim=-1, keepdim=True).expand_as(self.xyz_grad)
        self.features_dc_grad = self.features_dc_grad.mean(dim=-1, keepdim=True).expand_as(self.features_dc_grad)
        self.features_rest_grad = self.features_rest_grad.mean(dim=-1, keepdim=True).expand_as(self.features_rest_grad)
        self.scaling_grad = self.scaling_grad.mean(dim=-1, keepdim=True).expand_as(self.scaling_grad)
        self.rotation_grad = self.rotation_grad.mean(dim=-1, keepdim=True).expand_as(self.rotation_grad)
        self.opacity_grad = self.opacity_grad.mean(dim=-1, keepdim=True).expand_as(self.opacity_grad)
        self.exposure_grad = self.exposure_grad.mean(dim=-1, keepdim=True).expand_as(self.exposure_grad)


    @property
    def length(self):
        """
        Returns the length of the generalized vector
        """
        return (self.xyz_grad.numel() + self.features_dc_grad.numel() +
                self.features_rest_grad.numel() + self.scaling_grad.numel() +
                self.rotation_grad.numel() + self.opacity_grad.numel() +
                self.exposure_grad.numel())

    def load_1d_tensor(self, T):
        """
        Creates a GaussianModelState from a flattened tensor
        """
        assert T.shape[0] == self.length, "Input tensor must match the length of the model state"
        N1 = self.xyz_grad.numel()
        N2 = self.features_dc_grad.numel()
        N3 = self.features_rest_grad.numel()
        N4 = self.scaling_grad.numel()
        N5 = self.rotation_grad.numel()
        N6 = self.opacity_grad.numel()
        N7 = self.exposure_grad.numel()

        offset = 0
        xyz_grad = T[offset:offset + N1].view(self.xyz_grad.shape)
        offset += N1
        features_dc_grad = T[offset:offset + N2].view(self.features_dc_grad.shape)
        offset += N2
        features_rest_grad = T[offset:offset + N3].view(self.features_rest_grad.shape)
        offset += N3
        scaling_grad = T[offset:offset + N4].view(self.scaling_grad.shape)
        offset += N4
        rotation_grad = T[offset:offset + N5].view(self.rotation_grad.shape)
        offset += N5
        opacity_grad = T[offset:offset + N6].view(self.opacity_grad.shape)
        offset += N6
        exposure_grad = T[offset:offset + N7].view(self.exposure_grad.shape)

        self.__init__(xyz_grad, features_dc_grad, features_rest_grad,
                      scaling_grad, rotation_grad,
                      opacity_grad, exposure_grad)

    def as_1d_tensor(self):
        """
        Returns the model state as a flattened vector
        """
        return torch.cat((self.xyz_grad.flatten(),
                          self.features_dc_grad.flatten(),
                          self.features_rest_grad.flatten(),
                          self.scaling_grad.flatten(),
                          self.rotation_grad.flatten(),
                          self.opacity_grad.flatten(),
                          self.exposure_grad.flatten()), dim=0)

    def index_to_desc(self, index):
        N1 = self.xyz_grad.numel()
        N2 = self.features_dc_grad.numel()
        N3 = self.features_rest_grad.numel()
        N4 = self.scaling_grad.numel()
        N5 = self.rotation_grad.numel()
        N6 = self.opacity_grad.numel()
        N7 = self.exposure_grad.numel()

        assert index < self.length, "Index out of bounds for GaussianModelState"

        def find_coord(offset, shape):
            coords = []
            for dim in reversed(shape):
                coords.append(offset % dim)
                offset //= dim
            return tuple(reversed(coords))

        name_offsets = [(N1, "xyz_grad"), (N2, "features_dc_grad"), (N3, "features_rest_grad"),
                        (N4, "scaling_grad"), (N5, "rotation_grad"),
                        (N6, "opacity_grad"), (N7, "exposure_grad")]
        
        for l, name in name_offsets:
            if index < l:
                offset = index
                return name, find_coord(offset, getattr(self, name).shape)
            index -= l

    def __neg__(self):
        return GaussianModelState(
            -self.xyz_grad,
            -self.features_dc_grad,
            -self.features_rest_grad,
            -self.scaling_grad,
            -self.rotation_grad,
            -self.opacity_grad,
            -self.exposure_grad
        )

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelState(
                self.xyz_grad + other,
                self.features_dc_grad + other,
                self.features_rest_grad + other,
                self.scaling_grad + other,
                self.rotation_grad + other,
                self.opacity_grad + other,
                self.exposure_grad + other
            )
        elif isinstance(other, GaussianModelState):
            return GaussianModelState(
                self.xyz_grad + other.xyz_grad,
                self.features_dc_grad + other.features_dc_grad,
                self.features_rest_grad + other.features_rest_grad,
                self.scaling_grad + other.scaling_grad,
                self.rotation_grad + other.rotation_grad,
                self.opacity_grad + other.opacity_grad,
                self.exposure_grad + other.exposure_grad
            )
        else:
            raise TypeError(f"Can only add scalar values or GaussianModelState, not {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelState(
                self.xyz_grad - other,
                self.features_dc_grad - other,
                self.features_rest_grad - other,
                self.scaling_grad - other,
                self.rotation_grad - other,
                self.opacity_grad - other,
                self.exposure_grad - other
            )
        elif isinstance(other, GaussianModelState):
            return GaussianModelState(
                self.xyz_grad - other.xyz_grad,
                self.features_dc_grad - other.features_dc_grad,
                self.features_rest_grad - other.features_rest_grad,
                self.scaling_grad - other.scaling_grad,
                self.rotation_grad - other.rotation_grad,
                self.opacity_grad - other.opacity_grad,
                self.exposure_grad - other.exposure_grad
            )
        else:
            raise TypeError(f"Can only subtract scalar values or GaussianModelState, not {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelState(
                self.xyz_grad * other,
                self.features_dc_grad * other,
                self.features_rest_grad * other,
                self.scaling_grad * other,
                self.rotation_grad * other,
                self.opacity_grad * other,
                self.exposure_grad * other
            )
        elif isinstance(other, GaussianModelState):
            return GaussianModelState(
                self.xyz_grad * other.xyz_grad,
                self.features_dc_grad * other.features_dc_grad,
                self.features_rest_grad * other.features_rest_grad,
                self.scaling_grad * other.scaling_grad,
                self.rotation_grad * other.rotation_grad,
                self.opacity_grad * other.opacity_grad,
                self.exposure_grad * other.exposure_grad
            )
        elif isinstance(other, GaussianModelScaleMatrix):
            return GaussianModelState(
                self.xyz_grad * other.xyz_scale,
                self.features_dc_grad * other.features_dc_scale,
                self.features_rest_grad * other.features_rest_scale,
                self.scaling_grad * other.scaling_scale,
                self.rotation_grad * other.rotation_scale,
                self.opacity_grad * other.opacity_scale,
                self.exposure_grad * other.exposure_scale
            )
        else:
            raise TypeError(f"Can only multiply by scalar values, not {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelState(
                self.xyz_grad / other,
                self.features_dc_grad / other,
                self.features_rest_grad / other,
                self.scaling_grad / other,
                self.rotation_grad / other,
                self.opacity_grad / other,
                self.exposure_grad / other
            )
        elif isinstance(other, GaussianModelState):
            return GaussianModelState(
                self.xyz_grad / other.xyz_grad,
                self.features_dc_grad / other.features_dc_grad,
                self.features_rest_grad / other.features_rest_grad,
                self.scaling_grad / other.scaling_grad,
                self.rotation_grad / other.rotation_grad,
                self.opacity_grad / other.opacity_grad,
                self.exposure_grad / other.exposure_grad
            )
        elif isinstance(other, GaussianModelScaleMatrix):
            return GaussianModelState(
                self.xyz_grad / other.xyz_scale,
                self.features_dc_grad / other.features_dc_scale,
                self.features_rest_grad / other.features_rest_scale,
                self.scaling_grad / other.scaling_scale,
                self.rotation_grad / other.rotation_scale,
                self.opacity_grad / other.opacity_scale,
                self.exposure_grad / other.exposure_scale
            )
        else:
            raise TypeError(f"Can only divide by scalar values, not {type(other)}")

    def dot(self, other, damp=1):
        if isinstance(damp, (int, float)):
            s = torch.sum(self.xyz_grad * other.xyz_grad) + \
                torch.sum(self.features_dc_grad * other.features_dc_grad) + \
                torch.sum(self.features_rest_grad * other.features_rest_grad) + \
                torch.sum(self.scaling_grad * other.scaling_grad) + \
                torch.sum(self.rotation_grad * other.rotation_grad) + \
                torch.sum(self.opacity_grad * other.opacity_grad) + \
                torch.sum(self.exposure_grad * other.exposure_grad)
            s *= damp
        elif isinstance(damp, GaussianModelScaleMatrix):
            s = damp.xyz_scale * torch.sum(self.xyz_grad * other.xyz_grad) + \
                damp.features_dc_scale * torch.sum(self.features_dc_grad * other.features_dc_grad) + \
                damp.features_rest_scale * torch.sum(self.features_rest_grad * other.features_rest_grad) + \
                damp.scaling_scale * torch.sum(self.scaling_grad * other.scaling_grad) + \
                damp.rotation_scale * torch.sum(self.rotation_grad * other.rotation_grad) + \
                damp.opacity_scale * torch.sum(self.opacity_grad * other.opacity_grad) + \
                damp.exposure_scale * torch.sum(self.exposure_grad * other.exposure_grad)
        else:
            raise TypeError(f"damp must be a scalar or GaussianModelScaleMatrix, not {type(damp)}")

        return s.item()

    def sqrt(self):
        return GaussianModelState(
            torch.sqrt(self.xyz_grad),
            torch.sqrt(self.features_dc_grad),
            torch.sqrt(self.features_rest_grad),
            torch.sqrt(self.scaling_grad),
            torch.sqrt(self.rotation_grad),
            torch.sqrt(self.opacity_grad),
            torch.sqrt(self.exposure_grad)
        )
     
    def abs(self):
        return GaussianModelState(
            torch.abs(self.xyz_grad),
            torch.abs(self.features_dc_grad),
            torch.abs(self.features_rest_grad),
            torch.abs(self.scaling_grad),
            torch.abs(self.rotation_grad),
            torch.abs(self.opacity_grad),
            torch.abs(self.exposure_grad)
        )
