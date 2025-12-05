import torch
import torch.autograd.forward_ad as fwAD
import itertools
import math

class BatchLossImageState:
    """
    Represents the loss of multiple images as a generalized vector.
    The memory layout is such that the losses of the same type across all images are contiguous.
    """

    def __init__(self, loss_image, sizes_list, has_depth):
        """
        loss_image is a (B, C, H, W) tensor where C is 6 if no depth loss and 7 if depth loss is included.
        C = 0:2 -> Ll1_per_pixel (3 channels)
        C = 3:5 -> ssim_loss_per_pixel (3 channels)
        C = 6   -> Ll1depth_per_pixel (1 channel) or absent if no depth loss
        H and W are the maximum height and width across the batch.
        sizes_list is a list of tuples (H_i, W_i) for each image in the batch.
        has_depth is a boolean indicating if depth loss is included.
        """
        self.has_loss_image = True
        self.loss_image = loss_image
        self.Ll1_per_pixel = loss_image[:, :3, :, :]  # (B, 3, H, W)
        self.ssim_loss_per_pixel = loss_image[:, 3:6, :, :] # (B, 3, H, W)
        self.Ll1depth_per_pixel = loss_image[:, 6:, :, :]
        self.Ll1_scalar = 0.5 * (torch.linalg.vector_norm(self.Ll1_per_pixel.flatten(), ord=2) ** 2)
        self.ssim_loss_scalar = 0.5 * (torch.linalg.vector_norm(self.ssim_loss_per_pixel.flatten(), ord=2) ** 2)
        self.Ll1depth_scalar = 0.5 * (torch.linalg.vector_norm(self.Ll1depth_per_pixel.flatten(), ord=2) ** 2)
        self.loss_scalar = self.Ll1_scalar + self.ssim_loss_scalar + self.Ll1depth_scalar
        self.sizes_list = sizes_list  # List of tuples (H, W) for each image
        self.has_depth = has_depth  # Boolean indicating if depth loss is included

    def load_1d_tensor(self, T):
        self.has_loss_image = True
        self.loss_image = T.view(self.loss_image.shape)
        self.Ll1_per_pixel = self.loss_image[:, :3, :, :]  # (B, 3, H, W)
        self.ssim_loss_per_pixel = self.loss_image[:, 3:6, :, :] # (B, 3, H, W)
        self.Ll1depth_per_pixel = self.loss_image[:, 6:, :, :]
        self.Ll1_scalar = torch.linalg.vector_norm(self.Ll1_per_pixel.flatten(), ord=2) ** 2
        self.ssim_loss_scalar = torch.linalg.vector_norm(self.ssim_loss_per_pixel.flatten(), ord=2) ** 2
        self.Ll1depth_scalar = torch.linalg.vector_norm(self.Ll1depth_per_pixel.flatten(), ord=2) ** 2
        self.loss_scalar = self.Ll1_scalar + self.ssim_loss_scalar + self.Ll1depth_scalar

    def as_1d_tensor(self):
        return self.loss_image.flatten()


    def remove_loss_image(self):
        """
        Remove loss image to save memory
        TODO: make this cleaner
        """
        self.has_loss_image = False
        B, C, H, W = self.loss_image.shape
        self.loss_image = torch.empty((0, C, H, W), device=self.loss_image.device)
        self.Ll1_per_pixel = self.loss_image[:, :3, :, :]
        self.ssim_loss_per_pixel = self.loss_image[:, 3:6, :, :]
        self.Ll1depth_per_pixel = self.loss_image[:, 6:, :, :]

        torch.cuda.empty_cache()

    def check_invariant(self):
        Ll1_clone = self.Ll1_per_pixel.clone()
        ssim_clone = self.ssim_loss_per_pixel.clone()
        Ll1depth_clone = self.Ll1depth_per_pixel.clone()
        for i, (H, W) in enumerate(self.sizes_list):
            Ll1_clone[i, :, :H, :W] = 0
            ssim_clone[i, :, :H, :W] = 0
            if self.has_depth:
                Ll1depth_clone[i, :, :H, :W] = 0

        assert torch.all(Ll1_clone == 0), "Ll1_per_pixel has values outside valid image regions"
        assert torch.all(ssim_clone == 0), "ssim_loss_per_pixel has values outside valid image regions"
        if self.has_depth:
            assert torch.all(Ll1depth_clone == 0), "Ll1depth_per_pixel has values outside valid image regions"


    @property
    def length(self):
        """
        Returns the length of the generalized vector
        """
        Ll1_numel = 0
        ssim_numel = 0
        depth_numel = 0
        for H, W in self.sizes_list:
            Ll1_numel += 3 * H * W  # 3 channels
            ssim_numel += 3 * H * W  # 3 channels
            depth_numel += H * W
        if not self.has_depth:
            depth_numel = 0
        return Ll1_numel + ssim_numel + depth_numel

    def index_to_desc(self, index):
        N1 = self.Ll1_per_pixel.numel()
        N2 = self.ssim_loss_per_pixel.numel()
        N3 = self.Ll1depth_per_pixel.numel()

        assert index < self.length, "Index out of bounds for BatchLossImageState"

        def find_coord(offset, shape):
            coords = []
            for dim in reversed(shape):
                coords.append(offset % dim)
                offset //= dim
            return tuple(reversed(coords))

        name_offsets = [(N1, "Ll1_per_pixel"), (N2, "ssim_loss_per_pixel"), (N3, "Ll1depth_per_pixel")]
        
        for l, name in name_offsets:
            if index < l:
                offset = index
                return name, find_coord(offset, getattr(self, name).shape)
            index -= l

    def unpack_dual(self):
        loss_image_primal, loss_image_tangent = fwAD.unpack_dual(self.loss_image)

        primal = BatchLossImageState(loss_image_primal, self.sizes_list, self.has_depth)
        tangent = BatchLossImageState(loss_image_tangent, self.sizes_list, self.has_depth)

        return primal, tangent

    def backward(self, v, retain_graph=False):
        assert isinstance(v, BatchLossImageState), "v must be an instance of BatchLossImageState"
        self.loss_image.backward(v.loss_image, retain_graph=retain_graph)

    def zero_like(self):
        return BatchLossImageState(
            torch.zeros_like(self.loss_image),
            self.sizes_list,
            self.has_depth)

    def __add__(self, other):
        raise NotImplementedError("Addition of BatchLossImageState is not supported because we want to add the scalar losses not the vector loss")
        return BatchLossImageState(
            self.loss_image + other.loss_image,
            self.sizes_list,
            self.has_depth)

    def __sub__(self, other):
        raise NotImplementedError("Subtraction of BatchLossImageState is not supported because we want to subtract the scalar losses not the vector loss")
        return BatchLossImageState(
            self.loss_image - other.loss_image,
            self.sizes_list,
            self.has_depth)

    def __mul__(self, other):
        # Here we need to take the square root of other because we want the scalar loss to be multiplied by other
        if isinstance(other, (int, float)):
            return BatchLossImageState(
                self.loss_image * math.sqrt(other),
                self.sizes_list,
                self.has_depth)
        else:
            raise TypeError("Can only multiply by scalar values")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Here we need to take the square root of other because we want the scalar loss to be divided by other
        if isinstance(other, (int, float)):
            return BatchLossImageState(
                self.loss_image / math.sqrt(other),
                self.sizes_list,
                self.has_depth)
        else:
            raise TypeError("Can only divide by scalar values")

    def dot(self, other, damp):
        raise NotImplementedError("Dot product of BatchLossImageState is not supported. Just use scalar losses directly.")
        if isinstance(damp, (int, float)):
            s = torch.sum(self.loss_image * other.loss_image)
            s *= damp
        else:
            raise TypeError("Damping factor must be a scalar")
        return s.item()

class MultiBatchLossImageState:
    """
    Represents multiple batch losses as a generalized vector.
    """
    def __init__(self, batch_losses):
        self.batch_losses = batch_losses
        self.Ll1_scalar = sum(loss.Ll1_scalar for loss in self.batch_losses)
        self.ssim_loss_scalar = sum(loss.ssim_loss_scalar for loss in self.batch_losses)
        self.Ll1depth_scalar = sum(loss.Ll1depth_scalar for loss in self.batch_losses)
        self.loss_scalar = self.Ll1_scalar + self.ssim_loss_scalar + self.Ll1depth_scalar

    def check_invariant(self):
        for loss in self.batch_losses:
            loss.check_invariant()

    def load_1d_tensor(self, T):
        idx = 0
        for loss in self.batch_losses:
            N = loss.length
            loss_tensor = T[idx:idx + N]
            loss.load_1d_tensor(loss_tensor)
            idx += N

    def as_1d_tensor(self):
        return torch.cat([loss.as_1d_tensor() for loss in self.batch_losses], dim=0)


    @property
    def length(self):
        """
        Returns the length of the generalized vector
        """
        return sum(loss.length for loss in self.batch_losses)

    def unpack_dual(self):
        duals = [loss.unpack_dual() for loss in self.batch_losses]
        primals = [dual[0] for dual in duals]
        tangents = [dual[1] for dual in duals]
        primal = MultiBatchLossImageState(primals)
        tangent = MultiBatchLossImageState(tangents)
        return primal, tangent

    def backward(self, v, retain_graph=False):
        assert isinstance(v, MultiBatchLossImageState), "v must be an instance of MultiBatchLossImageState"
        for loss, v_loss in zip(self.batch_losses, v.batch_losses):
            loss.backward(v_loss, retain_graph=retain_graph)


    def zero_like(self):
        return MultiBatchLossImageState([loss.zero_like() for loss in self.batch_losses])

    def __add__(self, other):
        return MultiBatchLossImageState([loss + other_loss for loss, other_loss in zip(self.batch_losses, other.batch_losses)])

    def __sub__(self, other):
        return MultiBatchLossImageState([loss - other_loss for loss, other_loss in zip(self.batch_losses, other.batch_losses)])

    def __mul__(self, other):
        return MultiBatchLossImageState([loss * other for loss in self.batch_losses])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return MultiBatchLossImageState([loss / other for loss in self.batch_losses])

    def dot(self, other, damp):
        return sum(loss.dot(other_loss, damp) for loss, other_loss in zip(self.batch_losses, other.batch_losses))

