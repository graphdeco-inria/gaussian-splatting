import torch
import math
from utils.general_utils import safe_interact

class GaussianModelVector:
    """
    A vector representation for the parameters of a Gaussian model.
    For convenience, each parameter group can either be a vector or a scalar.
    A Gaussian model shall be used to initialize the GaussianModelVector to obtain 
    the correct shapes for each parameter group, in case we need to convert scalars
    to vectors.
    Operations on the GaussianModelVector are parameter-wise.
    """

    def __init__(self, xyz, features_dc, features_rest, scaling, rotation, opacity, exposure, 
                 gaussians=None, gaussian_model_vector=None,
                 xyz_shape=None, features_dc_shape=None,
                 features_rest_shape=None, scaling_shape=None,
                 rotation_shape=None, opacity_shape=None, exposure_shape=None):
        self.xyz = xyz
        self.features_dc = features_dc
        self.features_rest = features_rest
        self.scaling = scaling
        self.rotation = rotation
        self.opacity = opacity
        self.exposure = exposure

        if isinstance(self.xyz, torch.Tensor):
            self.tensor_xyz = True
            self.xyz_shape = self.xyz.shape
        else:
            self.tensor_xyz = False
            if xyz_shape is not None:
                self.xyz_shape = xyz_shape
            elif gaussians is not None:
                self.xyz_shape = gaussians._xyz.shape
            elif gaussian_model_vector is not None:
                self.xyz_shape = gaussian_model_vector.xyz_shape
            else:
                raise ValueError("Either gaussians or gaussian_model_vector must be provided to infer xyz_shape.")

        if isinstance(self.features_dc, torch.Tensor):
            self.tensor_features_dc = True
            self.features_dc_shape = self.features_dc.shape
        else:
            self.tensor_features_dc = False
            if features_dc_shape is not None:
                self.features_dc_shape = features_dc_shape
            elif gaussians is not None:
                self.features_dc_shape = gaussians._features_dc.shape
            elif gaussian_model_vector is not None:
                self.features_dc_shape = gaussian_model_vector.features_dc_shape
            else:
                raise ValueError("Either gaussians or gaussian_model_vector must be provided to infer features_dc_shape.")

        if isinstance(self.features_rest, torch.Tensor):
            self.tensor_features_rest = True
            self.features_rest_shape = self.features_rest.shape
        else:
            self.tensor_features_rest = False
            if features_rest_shape is not None:
                self.features_rest_shape = features_rest_shape
            elif gaussians is not None:
                self.features_rest_shape = gaussians._features_rest.shape
            elif gaussian_model_vector is not None:
                self.features_rest_shape = gaussian_model_vector.features_rest_shape
            else:
                raise ValueError("Either gaussians or gaussian_model_vector must be provided to infer features_rest_shape.")

        if isinstance(self.scaling, torch.Tensor):
            self.tensor_scaling = True
            self.scaling_shape = self.scaling.shape
        else:
            self.tensor_scaling = False
            if scaling_shape is not None:
                self.scaling_shape = scaling_shape
            elif gaussians is not None:
                self.scaling_shape = gaussians._scaling.shape
            elif gaussian_model_vector is not None:
                self.scaling_shape = gaussian_model_vector.scaling_shape
            else:
                raise ValueError("Either gaussians or gaussian_model_vector must be provided to infer scaling_shape.")

        if isinstance(self.rotation, torch.Tensor):
            self.tensor_rotation = True
            self.rotation_shape = self.rotation.shape
        else:
            self.tensor_rotation = False
            if rotation_shape is not None:
                self.rotation_shape = rotation_shape
            elif gaussians is not None:
                self.rotation_shape = gaussians._rotation.shape
            elif gaussian_model_vector is not None:
                self.rotation_shape = gaussian_model_vector.rotation_shape
            else:
                raise ValueError("Either gaussians or gaussian_model_vector must be provided to infer rotation_shape.")

        if isinstance(self.opacity, torch.Tensor):
            self.tensor_opacity = True
            self.opacity_shape = self.opacity.shape
        else:
            self.tensor_opacity = False
            if opacity_shape is not None:
                self.opacity_shape = opacity_shape
            elif gaussians is not None:
                self.opacity_shape = gaussians._opacity.shape
            elif gaussian_model_vector is not None:
                self.opacity_shape = gaussian_model_vector.opacity_shape
            else:
                raise ValueError("Either gaussians or gaussian_model_vector must be provided to infer opacity_shape.")

        if isinstance(self.exposure, torch.Tensor):
            self.tensor_exposure = True
            self.exposure_shape = self.exposure.shape
        else:
            self.tensor_exposure = False
            if exposure_shape is not None:
                self.exposure_shape = exposure_shape
            elif gaussians is not None:
                self.exposure_shape = gaussians._exposure.shape
            elif gaussian_model_vector is not None:
                self.exposure_shape = gaussian_model_vector.exposure_shape
            else:
                raise ValueError("Either gaussians or gaussian_model_vector must be provided to infer exposure_shape.")


    @classmethod
    def zeros_like(cls, other):
        from scene import GaussianModel
        if isinstance(other, GaussianModel):
            return cls(torch.zeros_like(other._xyz),
                       torch.zeros_like(other._features_dc),
                       torch.zeros_like(other._features_rest),
                       torch.zeros_like(other._scaling),
                       torch.zeros_like(other._rotation),
                       torch.zeros_like(other._opacity),
                       torch.zeros_like(other._exposure),)
        elif isinstance(other, GaussianModelVector):
            xyz = torch.zeros_like(other.xyz) if not other.scalar_xyz else 0.0
            features_dc = torch.zeros_like(other.features_dc) if not other.scalar_features_dc else 0.0
            features_rest = torch.zeros_like(other.features_rest) if not other.scalar_features_rest else 0.0
            scaling = torch.zeros_like(other.scaling) if not other.scalar_scaling else 0.0
            rotation = torch.zeros_like(other.rotation) if not other.scalar_rotation else 0.0
            opacity = torch.zeros_like(other.opacity) if not other.scalar_opacity else 0.0
            exposure = torch.zeros_like(other.exposure) if not other.scalar_exposure else 0.0
            return cls(xyz, features_dc, features_rest, scaling, rotation, opacity, exposure,
                       gaussians_model_vector=other)

    @classmethod
    def ones_like(cls, other):
        from scene import GaussianModel
        if isinstance(other, GaussianModel):
            return cls(torch.ones_like(other._xyz),
                       torch.ones_like(other._features_dc),
                       torch.ones_like(other._features_rest),
                       torch.ones_like(other._scaling),
                       torch.ones_like(other._rotation),
                       torch.ones_like(other._opacity),
                       torch.ones_like(other._exposure),)
        elif isinstance(other, GaussianModelVector):
            xyz = torch.ones_like(other.xyz) if not other.scalar_xyz else 1.0
            features_dc = torch.ones_like(other.features_dc) if not other.scalar_features_dc else 1.0
            features_rest = torch.ones_like(other.features_rest) if not other.scalar_features_rest else 1.0
            scaling = torch.ones_like(other.scaling) if not other.scalar_scaling else 1.0
            rotation = torch.ones_like(other.rotation) if not other.scalar_rotation else 1.0
            opacity = torch.ones_like(other.opacity) if not other.scalar_opacity else 1.0
            exposure = torch.ones_like(other.exposure) if not other.scalar_exposure else 1.0
            return cls(xyz, features_dc, features_rest, scaling, opacity, exposure,
                       gaussians_model_vector=other)

    @classmethod
    def rademacher_like(cls, other):
        def rademacher(shape):
            return torch.randint(0, 2, shape, device="cuda").float() * 2.0 - 1.0

        from scene import GaussianModel
        if isinstance(other, GaussianModel):
            return cls(rademacher(other._xyz.shape),
                       rademacher(other._features_dc.shape),
                       rademacher(other._features_rest.shape),
                       rademacher(other._scaling.shape),
                       rademacher(other._rotation.shape),
                       rademacher(other._opacity.shape),
                       rademacher(other._exposure.shape),)
        elif isinstance(other, GaussianModelVector):
            return cls(rademacher(other.xyz_shape),
                       rademacher(other.features_dc_shape),
                       rademacher(other.features_rest_shape),
                       rademacher(other.scaling_shape),
                       rademacher(other.rotation_shape),
                       rademacher(other.opacity_shape),
                       rademacher(other.exposure_shape),)

    @classmethod
    def from_gaussians_grad(cls, gaussians):
        xyz = gaussians._xyz.grad if gaussians._xyz.grad is not None else torch.zeros_like(gaussians._xyz)
        features_dc = gaussians._features_dc.grad if gaussians._features_dc.grad is not None else torch.zeros_like(gaussians._features_dc)
        features_rest = gaussians._features_rest.grad if gaussians._features_rest.grad is not None else torch.zeros_like(gaussians._features_rest)
        scaling = gaussians._scaling.grad if gaussians._scaling.grad is not None else torch.zeros_like(gaussians._scaling)
        rotation = gaussians._rotation.grad if gaussians._rotation.grad is not None else torch.zeros_like(gaussians._rotation)
        opacity = gaussians._opacity.grad if gaussians._opacity.grad is not None else torch.zeros_like(gaussians._opacity)
        exposure = gaussians._exposure.grad if gaussians._exposure.grad is not None else torch.zeros_like(gaussians._exposure)

        return cls(xyz, features_dc, features_rest, scaling, rotation, opacity, exposure,
                   gaussians=gaussians)

    def clone(self):
        return GaussianModelVector(
            self.xyz.clone() if self.tensor_xyz else self.xyz,
            self.features_dc.clone() if self.tensor_features_dc else self.features_dc,
            self.features_rest.clone() if self.tensor_features_rest else self.features_rest,
            self.scaling.clone() if self.tensor_scaling else self.scaling,
            self.rotation.clone() if self.tensor_rotation else self.rotation,
            self.opacity.clone() if self.tensor_opacity else self.opacity,
            self.exposure.clone() if self.tensor_exposure else self.exposure,
            gaussian_model_vector=self
        )

    def _get_param_group_lengths(self):
        N1 = math.prod(self.xyz_shape)
        N2 = math.prod(self.features_dc_shape)
        N3 = math.prod(self.features_rest_shape)
        N4 = math.prod(self.scaling_shape)
        N5 = math.prod(self.rotation_shape)
        N6 = math.prod(self.opacity_shape)
        N7 = math.prod(self.exposure_shape)
        return N1, N2, N3, N4, N5, N6, N7

    def load_1d_tensor(self, T, with_features_rest=True, with_exposure=True):
        N1, N2, N3, N4, N5, N6, N7 = self._get_param_group_lengths()

        check_len = N1 + N2 + N3 + N4 + N5 + N6 + N7
        if not with_features_rest:
            check_len -= N3
        if not with_exposure:
            check_len -= N7

        assert T.numel() == check_len, f"Expected tensor of length {check_len}, but got {T.numel()}"

        offset = 0
        self.xyz = T[offset:offset+N1].view(self.xyz_shape)
        offset += N1
        self.features_dc = T[offset:offset+N2].view(self.features_dc_shape)
        offset += N2
        if with_features_rest:
            self.features_rest = T[offset:offset+N3].view(self.features_rest_shape)
            offset += N3
        else:
            self.features_rest = 0.0
        self.scaling = T[offset:offset+N4].view(self.scaling_shape)
        offset += N4
        self.rotation = T[offset:offset+N5].view(self.rotation_shape)
        offset += N5
        self.opacity = T[offset:offset+N6].view(self.opacity_shape)
        offset += N6
        if with_exposure:
            self.exposure = T[offset:offset+N7].view(self.exposure_shape)
        else:
            self.exposure = 0.0

    def as_1d_tensor(self, with_features_rest=True, with_exposure=True):
        N1, N2, N3, N4, N5, N6, N7 = self._get_param_group_lengths()

        l = []
        if self.tensor_xyz:
            l.append(self.xyz.flatten())
        else:
            l.append(torch.ones(N1, device="cuda") * self.xyz)
        if self.tensor_features_dc:
            l.append(self.features_dc.flatten())
        else:
            l.append(torch.ones(N2, device="cuda") * self.features_dc)
        if with_features_rest:
            if self.tensor_features_rest:
                l.append(self.features_rest.flatten())
            else:
                l.append(torch.ones(N3, device="cuda") * self.features_rest)
        if self.tensor_scaling:
            l.append(self.scaling.flatten())
        else:
            l.append(torch.ones(N4, device="cuda") * self.scaling)
        if self.tensor_rotation:
            l.append(self.rotation.flatten())
        else:
            l.append(torch.ones(N5, device="cuda") * self.rotation)
        if self.tensor_opacity:
            l.append(self.opacity.flatten())
        else:
            l.append(torch.ones(N6, device="cuda") * self.opacity)
        if with_exposure:
            if self.tensor_exposure:
                l.append(self.exposure.flatten())
            else:
                l.append(torch.ones(N7, device="cuda") * self.exposure)

        return torch.cat(l, dim=0)


    def clip_(self, min_value, max_value):
        if isinstance(min_value, GaussianModelVector) or isinstance(max_value, GaussianModelVector):
            self.xyz.clip_(min_value.xyz, max_value.xyz)
            self.features_dc.clip_(min_value.features_dc, max_value.features_dc)
            self.features_rest.clip_(min_value.features_rest, max_value.features_rest)
            self.scaling.clip_(min_value.scaling, max_value.scaling)
            self.rotation.clip_(min_value.rotation, max_value.rotation)
            self.opacity.clip_(min_value.opacity, max_value.opacity)
            self.exposure.clip_(min_value.exposure, max_value.exposure)
        elif isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
            if self.tensor_xyz:
                self.xyz.clamp_(min_value, max_value)
            else:
                self.xyz = max(min(self.xyz, max_value), min_value)
            if self.tensor_features_dc:
                self.features_dc.clamp_(min_value, max_value)
            else:
                self.features_dc = max(min(self.features_dc, max_value), min_value)
            if self.tensor_features_rest:
                self.features_rest.clamp_(min_value, max_value)
            else:
                self.features_rest = max(min(self.features_rest, max_value), min_value)
            if self.tensor_scaling:
                self.scaling.clamp_(min_value, max_value)
            else:
                self.scaling = max(min(self.scaling, max_value), min_value)
            if self.tensor_rotation:
                self.rotation.clamp_(min_value, max_value)
            else:
                self.rotation = max(min(self.rotation, max_value), min_value)
            if self.tensor_opacity:
                self.opacity.clamp_(min_value, max_value)
            else:
                self.opacity = max(min(self.opacity, max_value), min_value)
            if self.tensor_exposure:
                self.exposure.clamp_(min_value, max_value)
            else:
                self.exposure = max(min(self.exposure, max_value), min_value)

    def __neg__(self):
        return GaussianModelVector(
            -self.xyz,
            -self.features_dc,
            -self.features_rest,
            -self.scaling,
            -self.rotation,
            -self.opacity,
            -self.exposure,
            gaussian_model_vector=self
        )

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelVector(
                self.xyz + other,
                self.features_dc + other,
                self.features_rest + other,
                self.scaling + other,
                self.rotation + other,
                self.opacity + other,
                self.exposure + other,
                gaussian_model_vector=self
            )
        elif isinstance(other, GaussianModelVector):
            return GaussianModelVector(
                self.xyz + other.xyz,
                self.features_dc + other.features_dc,
                self.features_rest + other.features_rest,
                self.scaling + other.scaling,
                self.rotation + other.rotation,
                self.opacity + other.opacity,
                self.exposure + other.exposure,
                gaussian_model_vector=self
            )
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelVector(
                self.xyz - other,
                self.features_dc - other,
                self.features_rest - other,
                self.scaling - other,
                self.rotation - other,
                self.opacity - other,
                self.exposure - other,
                gaussian_model_vector=self
            )
        elif isinstance(other, GaussianModelVector):
            return GaussianModelVector(
                self.xyz - other.xyz,
                self.features_dc - other.features_dc,
                self.features_rest - other.features_rest,
                self.scaling - other.scaling,
                self.rotation - other.rotation,
                self.opacity - other.opacity,
                self.exposure - other.exposure,
                gaussian_model_vector=self
            )
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelVector(
                other - self.xyz,
                other - self.features_dc,
                other - self.features_rest,
                other - self.scaling,
                other - self.rotation,
                other - self.opacity,
                other - self.exposure,
                gaussian_model_vector=self
            )
        elif isinstance(other, GaussianModelVector):
            return GaussianModelVector(
                other.xyz - self.xyz,
                other.features_dc - self.features_dc,
                other.features_rest - self.features_rest,
                other.scaling - self.scaling,
                other.rotation - self.rotation,
                other.opacity - self.opacity,
                other.exposure - self.exposure,
                gaussian_model_vector=self
            )
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelVector(
                self.xyz * other,
                self.features_dc * other,
                self.features_rest * other,
                self.scaling * other,
                self.rotation * other,
                self.opacity * other,
                self.exposure * other,
                gaussian_model_vector=self
            )
        elif isinstance(other, GaussianModelVector):
            return GaussianModelVector(
                self.xyz * other.xyz,
                self.features_dc * other.features_dc,
                self.features_rest * other.features_rest,
                self.scaling * other.scaling,
                self.rotation * other.rotation,
                self.opacity * other.opacity,
                self.exposure * other.exposure,
                gaussian_model_vector=self
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelVector(
                other / self.xyz,
                other / self.features_dc,
                other / self.features_rest,
                other / self.scaling,
                other / self.rotation,
                other / self.opacity,
                other / self.exposure,
                gaussian_model_vector=self
            )
        elif isinstance(other, GaussianModelVector):
            return GaussianModelVector(
                other.xyz / self.xyz,
                other.features_dc / self.features_dc,
                other.features_rest / self.features_rest,
                other.scaling / self.scaling,
                other.rotation / self.rotation,
                other.opacity / self.opacity,
                other.exposure / self.exposure,
                gaussian_model_vector=self
            )

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return GaussianModelVector(
                self.xyz / other,
                self.features_dc / other,
                self.features_rest / other,
                self.scaling / other,
                self.rotation / other,
                self.opacity / other,
                self.exposure / other,
                gaussian_model_vector=self
            )
        elif isinstance(other, GaussianModelVector):
            return GaussianModelVector(
                self.xyz / other.xyz,
                self.features_dc / other.features_dc,
                self.features_rest / other.features_rest,
                self.scaling / other.scaling,
                self.rotation / other.rotation,
                self.opacity / other.opacity,
                self.exposure / other.exposure,
                gaussian_model_vector=self
            )

    def sum(self):
        N1, N2, N3, N4, N5, N6, N7 = self._get_param_group_lengths()

        s = 0.0
        if self.tensor_xyz:
            s += self.xyz.sum()
        else:
            s += self.xyz * N1
        if self.tensor_features_dc:
            s += self.features_dc.sum()
        else:
            s += self.features_dc * N2
        if self.tensor_features_rest:
            s += self.features_rest.sum()
        else:
            s += self.features_rest * N3
        if self.tensor_scaling:
            s += self.scaling.sum()
        else:
            s += self.scaling * N4
        if self.tensor_rotation:
            s += self.rotation.sum()
        else:
            s += self.rotation * N5
        if self.tensor_opacity:
            s += self.opacity.sum()
        else:
            s += self.opacity * N6
        if self.tensor_exposure:
            s += self.exposure.sum()
        else:
            s += self.exposure * N7

        if isinstance(s, torch.Tensor):
            return s.item()

        return s

    def dot(self, other):
        return (self * other).sum()

    def sqrt(self):
        return GaussianModelVector(
            torch.sqrt(self.xyz) if self.tensor_xyz else math.sqrt(self.xyz),
            torch.sqrt(self.features_dc) if self.tensor_features_dc else math.sqrt(self.features_dc),
            torch.sqrt(self.features_rest) if self.tensor_features_rest else math.sqrt(self.features_rest),
            torch.sqrt(self.scaling) if self.tensor_scaling else math.sqrt(self.scaling),
            torch.sqrt(self.rotation) if self.tensor_rotation else math.sqrt(self.rotation),
            torch.sqrt(self.opacity) if self.tensor_opacity else math.sqrt(self.opacity),
            torch.sqrt(self.exposure) if self.tensor_exposure else math.sqrt(self.exposure),
            gaussian_model_vector=self
        )

    def abs(self):
        return GaussianModelVector(
            torch.abs(self.xyz) if self.tensor_xyz else abs(self.xyz),
            torch.abs(self.features_dc) if self.tensor_features_dc else abs(self.features_dc),
            torch.abs(self.features_rest) if self.tensor_features_rest else abs(self.features_rest),
            torch.abs(self.scaling) if self.tensor_scaling else abs(self.scaling),
            torch.abs(self.rotation) if self.tensor_rotation else abs(self.rotation),
            torch.abs(self.opacity) if self.tensor_opacity else abs(self.opacity),
            torch.abs(self.exposure) if self.tensor_exposure else abs(self.exposure),
            gaussian_model_vector=self
        )


