import torch
from utils.general_utils import safe_interact

class Preconditioner:
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, vector):
        # Placeholder for preconditioning logic
        return vector  # No actual preconditioning applied

class AdaHessianPreconditioner:
    def __init__(self, z_gen_func, beta1=0.9, beta2=0.999, eps=1e-16, hessian_power=1.0):
        self.iteration = 0
        self.z_gen_func = z_gen_func
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.D_sq = 0 
        self.D = 0

        self.S = None
        self.damp = None

    def set_rescale_and_damp(self, S, damp):
        self.S = S
        self.damp = damp

    def reset(self):
        self.iteration = 0
        self.D_sq = 0
        self.D = 0
        self.S = None
        self.damp = None

    def update(self, Hz_func, cam_provider, scale, num_iter):
        self.iteration += 1

        D_sq_accum = 0
        for _ in range(num_iter):
            cam_provider.sample_new()
            vcs = cam_provider.get_cur_batch()
            z = self.z_gen_func()
            Di = z * Hz_func(z, viewpoint_cams=vcs, scale=scale)
            D_sq_accum = Di * Di + D_sq_accum
            del Di, z

        D_sq_accum = D_sq_accum / num_iter
        self.D_sq = (1 - self.beta2) * D_sq_accum + self.beta2 * self.D_sq

        # D_accum = 0
        # for _ in range(num_iter):
        #     cam_provider.sample_new()
        #     vcs = cam_provider.get_cur_batch()
        #     z = self.z_gen_func()
        #     Di = z * Hz_func(z, viewpoint_cams=vcs, scale=scale)
        #     D_accum = Di + D_accum
        #     del Di, z

        # D_accum = D_accum / num_iter
        # self.D = (1 - self.beta2) * D_accum + self.beta2 * self.D

    @property
    def D_corrected(self):
        return self.D_sq / (1 - self.beta2 ** self.iteration)

    def __call__(self, v):
        D_corrected_sqrt = (self.D_sq / (1 - self.beta2 ** self.iteration)).sqrt()
        return v / (D_corrected_sqrt + self.eps)

        # return v / (self.D.abs() + self.eps)

        # D_corrected = (self.D_sq / (1 - self.beta2 ** self.iteration)).sqrt()

        # if self.S is not None:
        #     D_corrected = self.S * D_corrected * self.S

        # if self.damp is not None:
        #     D_corrected = D_corrected + self.damp


        # D_corrected = self.D / (1 - self.beta2 ** self.iteration)
        # D_corrected = D_corrected.abs()

        # if self.S is not None:
        #     D_corrected = self.S * D_corrected * self.S

        # if self.damp is not None:
        #     D_corrected = D_corrected + self.damp

        # safe_interact(local=locals(), banner="In AdaHessian preconditioner call")

        return v / (D_corrected + self.eps)

class ConstantPreconditioner:
    def __init__(self, constant, eps=1e-16):
        self.constant = constant
        self.eps = eps

    def reset(self):
        pass

    def update(self, Hz_func, cam_provider, scale, num_iter):
        pass

    def __call__(self, v):
        import code; code.interact(local=locals(), banner="in constant preconditioner call")
        return v / (self.constant + self.eps)
