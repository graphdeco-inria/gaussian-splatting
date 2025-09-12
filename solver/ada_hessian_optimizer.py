import torch
from utils.general_utils import safe_interact

class AdaHessianOptimizer:
    def __init__(self, z_gen_func, beta1=0.9, beta2=0.999, eps=1e-8, hessian_power=1.0):
        self.iteration = 0
        self.z_gen_func = z_gen_func
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.g = 0
        self.D_sq = 0 

    def reset(self):
        self.iteration = 0
        self.D_sq = 0

    def get_update_step(self, Hz_func, g_func, cam_provider, scale, num_iter):
        self.iteration += 1
        g_accum = 0
        D_accum = 0
        with torch.no_grad():
            for _ in range(num_iter):
                print("AdaHessian iteration ", self.iteration, " sub-iter ", _+1, "/", num_iter)
                cam_provider.sample_new()
                vcs = cam_provider.get_cur_batch()
                gi, _ = g_func(viewpoint_cams=vcs, scale=scale)
                g_accum = gi + g_accum
                z = self.z_gen_func()
                Di = z * Hz_func(z, viewpoint_cams=vcs, scale=scale)
                D_accum = Di * Di + D_accum
                del gi, Di, z

        g_accum = g_accum / num_iter
        self.g = ((1 - self.beta1) / num_iter) * g_accum + self.beta1 * self.g

        D_accum = D_accum / num_iter
        D_accum.block_average_and_expand()
        self.D_sq = ((1 - self.beta2) / num_iter) * D_accum + self.beta2 * self.D_sq

        g_corrected = self.g / (1 - self.beta1 ** self.iteration)
        D_sq_corrected = self.D_sq / (1 - self.beta2 ** self.iteration)

        step = -g_corrected / (D_sq_corrected.sqrt() + self.eps)

        return step
