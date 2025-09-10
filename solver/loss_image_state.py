import torch
import torch.autograd.forward_ad as fwAD
import itertools

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
        self.Ll1_scalar = torch.linalg.vector_norm(self.Ll1_per_pixel.flatten(), ord=2) ** 2
        self.ssim_loss_scalar = torch.linalg.vector_norm(self.ssim_loss_per_pixel.flatten(), ord=2) ** 2
        self.Ll1depth_scalar = torch.linalg.vector_norm(self.Ll1depth_per_pixel.flatten(), ord=2) ** 2
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
        return BatchLossImageState(
            self.loss_image + other.loss_image,
            self.sizes_list,
            self.has_depth)

    def __sub__(self, other):
        return BatchLossImageState(
            self.loss_image - other.loss_image,
            self.sizes_list,
            self.has_depth)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return BatchLossImageState(
                self.loss_image * other,
                self.sizes_list,
                self.has_depth)
        else:
            raise TypeError("Can only multiply by scalar values")

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other, damp):
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

    def dot(self, other, damp):
        return sum(loss.dot(other_loss, damp) for loss, other_loss in zip(self.batch_losses, other.batch_losses))

class LossImageState:
    """
    Represents the loss as a generalized vector.
    """

    def __init__(self, Ll1_per_pixel, ssim_loss_per_pixel, Ll1depth_per_pixel):
        self.Ll1_per_pixel = Ll1_per_pixel
        self.ssim_loss_per_pixel = ssim_loss_per_pixel
        self.Ll1depth_per_pixel = Ll1depth_per_pixel
        self.Ll1_scalar = torch.linalg.vector_norm(Ll1_per_pixel.flatten(), ord=2) ** 2
        self.ssim_loss_scalar = torch.linalg.vector_norm(ssim_loss_per_pixel.flatten(), ord=2) ** 2
        self.Ll1depth_scalar = torch.linalg.vector_norm(Ll1depth_per_pixel.flatten(), ord=2) ** 2
        self.loss_scalar = self.Ll1_scalar + self.ssim_loss_scalar + self.Ll1depth_scalar
        assert self.Ll1depth_per_pixel is not None, "Ll1depth_per_pixel must not be None"

    @property
    def length(self):
        """
        Returns the length of the generalized vector
        """
        return self.Ll1_per_pixel.numel() + self.ssim_loss_per_pixel.numel() + self.Ll1depth_per_pixel.numel()

    def load_1d_tensor(self, T):
        """
        Creates a LossImageState from a flattened tensor
        """
        assert T.shape[0] == self.length, "Input tensor must match the length of the loss state"
        N1, N2, N3 = self.Ll1_per_pixel.numel(), self.ssim_loss_per_pixel.numel(), self.Ll1depth_per_pixel.numel()
        Ll1_per_pixel = T[:N1].view(self.Ll1_per_pixel.shape)
        ssim_loss_per_pixel = T[N1:N1 + N2].view(self.ssim_loss_per_pixel.shape)
        Ll1depth_per_pixel = T[N1 + N2:N1 + N2 + N3].view(self.Ll1depth_per_pixel.shape)
        
        self.__init__(Ll1_per_pixel, ssim_loss_per_pixel, Ll1depth_per_pixel)
        

    def as_1d_tensor(self):
        """
        Returns the loss as a flattened vector
        """
        return torch.cat((self.Ll1_per_pixel.flatten(),
                          self.ssim_loss_per_pixel.flatten(),
                          self.Ll1depth_per_pixel.flatten()), dim=0)

    def index_to_desc(self, index):
        N1 = self.Ll1_per_pixel.numel()
        N2 = self.ssim_loss_per_pixel.numel()
        N3 = self.Ll1depth_per_pixel.numel()

        assert index < self.length, "Index out of bounds for LossImageState"

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
        L1_primal, L1_tangent = fwAD.unpack_dual(self.Ll1_per_pixel)
        ssim_primal, ssim_tangent = fwAD.unpack_dual(self.ssim_loss_per_pixel)
        Ll1depth_primal, Ll1depth_tangent = fwAD.unpack_dual(self.Ll1depth_per_pixel)

        L1_tangent = L1_tangent if L1_tangent is not None else torch.zeros_like(L1_primal)
        ssim_tangent = ssim_tangent if ssim_tangent is not None else torch.zeros_like(ssim_primal)
        Ll1depth_tangent = Ll1depth_tangent if Ll1depth_tangent is not None else torch.zeros_like(Ll1depth_primal)

        primal = LossImageState(L1_primal, ssim_primal, Ll1depth_primal)
        tangent = LossImageState(L1_tangent, ssim_tangent, Ll1depth_tangent)

        return primal, tangent

    def backward(self, v, retain_graph=False):
        assert isinstance(v, LossImageState), "v must be an instance of LossImageState"
        self.Ll1_per_pixel.backward(v.Ll1_per_pixel, retain_graph=retain_graph)
        self.ssim_loss_per_pixel.backward(v.ssim_loss_per_pixel, retain_graph=retain_graph)
        self.Ll1depth_per_pixel.backward(v.Ll1depth_per_pixel, retain_graph=retain_graph)

    def zero_like(self):
        return LossImageState(
            torch.zeros_like(self.Ll1_per_pixel),
            torch.zeros_like(self.ssim_loss_per_pixel),
            torch.zeros_like(self.Ll1depth_per_pixel)
        )

    def __add__(self, other):
        return LossImageState(
            self.Ll1_per_pixel + other.Ll1_per_pixel,
            self.ssim_loss_per_pixel + other.ssim_loss_per_pixel,
            self.Ll1depth_per_pixel + other.Ll1depth_per_pixel
        )

    def __sub__(self, other):
        return LossImageState(
            self.Ll1_per_pixel - other.Ll1_per_pixel,
            self.ssim_loss_per_pixel - other.ssim_loss_per_pixel,
            self.Ll1depth_per_pixel - other.Ll1depth_per_pixel
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return LossImageState(
                self.Ll1_per_pixel * other,
                self.ssim_loss_per_pixel * other,
                self.Ll1depth_per_pixel * other
            )
        else:
            raise TypeError("Can only multiply by scalar values")

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other, damp):
        if isinstance(damp, (int, float)):
            s = torch.sum(self.Ll1_per_pixel * other.Ll1_per_pixel) + \
                torch.sum(self.ssim_loss_per_pixel * other.ssim_loss_per_pixel) + \
                torch.sum(self.Ll1depth_per_pixel * other.Ll1depth_per_pixel)
            s *= damp
        else:
            raise TypeError("Damping factor must be a scalar")
        return s.item()  # Return a scalar value
     

class MultiLossImageState:
    """
    Represents the loss of multiple images as a generalized vector.
    """
    def __init__(self, loss_states):
        self.loss_states = loss_states

        for loss_state in self.loss_states:
            assert isinstance(loss_state, LossImageState), "Each loss state must be an instance of LossImageState" 

        self.Ll1_scalar = sum(loss_state.Ll1_scalar for loss_state in self.loss_states)
        self.ssim_loss_scalar = sum(loss_state.ssim_loss_scalar for loss_state in self.loss_states)
        self.Ll1depth_scalar = sum(loss_state.Ll1depth_scalar for loss_state in self.loss_states)
        self.loss_scalar = self.Ll1_scalar + self.ssim_loss_scalar + self.Ll1depth_scalar

    @property
    def length(self):
        """
        Returns the length of the generalized vector
        """
        return sum(loss_state.length for loss_state in self.loss_states)

    def load_1d_tensor(self, T):
        """
        Creates a LossImageState from a flattened tensor
        """
        assert T.shape[0] == self.length, "Input tensor must match the length of the loss state"
        offset = 0
        loss_states = []
        for loss_state in self.loss_states:
            N = loss_state.length
            loss_state_tensor = T[offset:offset + N]
            loss_state.load_1d_tensor(loss_state_tensor)
            loss_states.append(loss_state)
            offset += N
        self.__init__(loss_states)
        

    def as_1d_tensor(self):
        """
        Returns the loss as a flattened vector
        """
        return torch.cat([loss_state.as_1d_tensor() for loss_state in self.loss_states], dim=0)

    def index_to_desc(self, index):
        for i in range(len(self.loss_states)):
            loss_state = self.loss_states[i]
            if index < loss_state.length:
                t = loss_state.index_to_desc(index)
                return f"Image {i}: " + t[0] + f" at {t[1]}"
            index -= loss_state.length

    def unpack_dual(self):
        duals = [loss_state.unpack_dual() for loss_state in self.loss_states]
        primals = [dual[0] for dual in duals]
        tangents = [dual[1] for dual in duals]
        return MultiLossImageState(primals), MultiLossImageState(tangents)

    def backward(self, v, retain_graph=False):
        assert isinstance(v, MultiLossImageState), "v must be an instance of MultiLossImageState"
        for loss_state, v_state in zip(self.loss_states, v.loss_states):
            loss_state.backward(v_state, retain_graph=retain_graph)


    def zero_like(self):
        return MultiLossImageState([loss_state.zero_like() for loss_state in self.loss_states])

    def __add__(self, other):
        return MultiLossImageState([loss_state + other_loss_state for loss_state, other_loss_state in zip(self.loss_states, other.loss_states)])

    def __sub__(self, other):
        return MultiLossImageState([loss_state - other_loss_state for loss_state, other_loss_state in zip(self.loss_states, other.loss_states)])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MultiLossImageState([loss_state * other for loss_state in self.loss_states])
        else:
            raise TypeError("Can only multiply by scalar values")

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other, damp):
        return sum(loss_state.dot(other_loss_state, damp) for loss_state, other_loss_state in zip(self.loss_states, other.loss_states))
