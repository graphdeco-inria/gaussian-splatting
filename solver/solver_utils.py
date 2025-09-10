import numpy as np
import torch

class CamProvider:
    """
    We have 3 use cases for the camera provider:
    1. A random small batch of cameras for updating the preconditioner or momentum-based optimizer. This batch can change every time.
        2 sub modes
        a. Go through all cameras one by one, this is needed because we want all Guassians to be seen
        b. Random cameras with a random stride, but we want this random batch comes from the cameras PCG is working on
    2. A fixed batch of cameras for PCG. This batch must remain the same for the entire PCG run. 
    3. A fixed batch of validation cameras for loss evaluation.
    Use advance to get the next batch of cameras. Use cur_batch to get the current batch of cameras.
    """

    def __init__(self, viewpoint_cams, mode, max_stride=1, sample_size=1):
        self.viewpoint_cams = viewpoint_cams
        self.mode = mode
        self.max_stride = max_stride
        self.B = len(viewpoint_cams)
        self.sample_size = sample_size if sample_size > 0 else self.B
        self.start_idx = 0
        self.cur_samples = None

    def sample_new(self):
        if self.mode == "all":
            rand_indices = list(range(self.B))
        elif self.mode == "one-by-one":
            rand_indices = [(self.start_idx + i * self.max_stride) % self.B for i in range(self.sample_size)]
            self.start_idx = self.start_idx + self.max_stride * self.sample_size
        elif self.mode == "step":
            self.start_idx = np.random.randint(0, self.B)
            self.start_idx = self.start_idx
            rand_indices = []
            for _ in range(self.sample_size):
                rand_indices.append(self.start_idx)
                self.start_idx = (self.start_idx + np.random.randint(1, self.max_stride)) % self.B
        elif self.mode == "strided":
            rand_indices = [self.viewpoint_cams[(self.start_idx + i * self.max_stride) % self.B] for i in range(self.sample_size)]
            self.start_idx = self.start_idx + self.sample_size * self.max_stride
        elif self.mode == "random":
            rand_indices = np.random.choice(self.B, self.sample_size, replace=False)
        else:
            raise ValueError(f"Unknown camera provider mode: {self.mode}")

        self.cur_batch = [self.viewpoint_cams[i] for i in rand_indices]

    def get_cur_batch(self):
        return self.cur_batch
