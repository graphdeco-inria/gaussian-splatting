import torch
from solver.diagonal_estimator import restarted_hutchinson

class SophiaOptimizer:
    def __init__(self, lr=1.0, betas=(0.9, 0.99), eps=1e-15, clip=False, 
                 adam_beta2=0.999,
                 gamma=1.0,
                 diagonal_update_interval=10,
                 num_init_iter=20, 
                 num_init_restart_iter=3, 
                 num_update_iter=2, 
                 num_update_restart_iter=1):
        self.lr_init = lr
        self.betas = betas
        self.gamma = gamma
        self.eps = eps
        self.clip = clip
        self.diagonal_update_interval = diagonal_update_interval
        self.num_init_iter = num_init_iter
        self.num_init_restart_iter = num_init_restart_iter
        self.num_update_iter = num_update_iter
        self.num_update_restart_iter = num_update_restart_iter

        self.reset()

    def set_clip(self, clip):
        self.clip = clip

    def reset(self):
        self.iter = 0
        self.m = 0
        self.lr = self.lr_init
        self.diagonal_initialized = False
        self.D_iter = 0
        self.D_smoothed = 0
        self.D_est = 0

    def update_lr(self, lr):
        self.lr = lr

    def densify_and_prune(self, prune_mask):
        if self.iter == 0:
            return
        self.m.densify_and_prune_(prune_mask)
        self.D_smoothed.densify_and_prune_(prune_mask)
        self.D_est.densify_and_prune_(prune_mask)

    def get_update(self, g, JTJv_func, Dhat_func, z_gen_func, S):
        if self.diagonal_initialized:
            self._handle_new_parameters(g)
        if self.iter % self.diagonal_update_interval == 0 or not self.diagonal_initialized:
            self.update_diagonal(JTJv_func, Dhat_func, z_gen_func, S)


        self.iter += 1
        bias_correction1 = 1 - self.betas[0] ** self.iter

        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * g

        v = self.gamma * self.D_est

        m_hat = self.m / bias_correction1

        s = -m_hat / (v + self.eps)

        if self.clip:
            s.clip_(-self.lr, self.lr)

        return s

    def _handle_new_parameters(self, g):
        g_vec = g.as_1d_tensor()
        D_smoothed_vec = self.D_smoothed.as_1d_tensor()
        new_params_mask = (D_smoothed_vec == 0.0) & (g_vec != 0.0)
        if new_params_mask.any():
            beta2 = self.betas[1]
            # print(f"New parameters detected, initializing their D_smoothed values.")
            D_smoothed_vec[new_params_mask] = g_vec[new_params_mask].abs()
            self.D_smoothed.load_1d_tensor(D_smoothed_vec)
            self.D_est = self.D_smoothed / (1 - beta2 ** self.D_iter)

    def update_diagonal(self, JTJv_func, Dhat_func, z_gen_func, S):
        if not self.diagonal_initialized:
            self.diagonal_initialized = True
            num_diag_iter = self.num_init_iter
            restart_iter = self.num_init_restart_iter
            D_init = Dhat_func()
        else:
            num_diag_iter = self.num_update_iter
            restart_iter = self.num_update_restart_iter
            D_init = self.D_est

        D_est_t = restarted_hutchinson(Hz_func=JTJv_func,
                                       z_gen_func=z_gen_func,
                                       D_init=D_init,
                                       restart_iter=restart_iter,
                                       num_iters=num_diag_iter,
                                       )
        D_est_t = D_est_t.abs() / (S * S)

        beta2 = self.betas[1]
        self.D_iter += 1
        self.D_smoothed = beta2 * self.D_smoothed + (1 - beta2) * D_est_t
        self.D_est = self.D_smoothed / (1 - beta2 ** self.D_iter)

