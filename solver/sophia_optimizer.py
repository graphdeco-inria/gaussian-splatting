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
        self.total_D_iter = 0
        self.g_iter_vec = 0
        self.m = 0
        self.lr = self.lr_init
        self.diagonal_initialized = False
        self.D_iter_vec = 0
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
        else:
            # Initialize states
            self.g_iter_vec = torch.zeros(g.as_1d_tensor().shape, device=g.xyz.device, dtype=torch.long)
            self.m = g * 0.0
            self.D_iter_vec = torch.zeros_like(self.g_iter_vec)
            self.D_smoothed = g * 0.0
            self.D_est = g * 0.0
        if self.iter % self.diagonal_update_interval == 0 or not self.diagonal_initialized:
            self.update_diagonal(JTJv_func, Dhat_func, z_gen_func, S)

        self.iter += 1

        # This makes the optimizer implementation not generalizable, but it works for now
        g_vec = g.as_1d_tensor()
        m_vec = self.m.as_1d_tensor()
        nonzero_indices = g_vec.nonzero(as_tuple=True)[0]
        
        self.g_iter_vec[nonzero_indices] += 1

        m_vec[nonzero_indices] = self.betas[0] * m_vec[nonzero_indices] + (1 - self.betas[0]) * g_vec[nonzero_indices]
        m_hat_vec = torch.zeros_like(m_vec)
        m_hat_vec = m_vec / (1 - (self.betas[0] ** self.g_iter_vec) + self.eps) * (self.g_iter_vec / self.iter) # Need to normalize at the end

        v_vec = self.gamma * self.D_est.as_1d_tensor()
        # s_vec = -m_hat_vec / (v_vec + self.eps)
        s_vec = torch.zeros_like(m_vec)
        s_vec[nonzero_indices] = -m_hat_vec[nonzero_indices] / (v_vec[nonzero_indices] + self.eps)
        s = g * 0.0
        s.load_1d_tensor(s_vec)

        # self.m = self.betas[0] * self.m + (1 - self.betas[0]) * g
        # v = self.gamma * self.D_est
        # m_hat = self.m / (1 - self.betas[0] ** self.g_iter)
        # s = -m_hat / (v + self.eps)

        if self.clip:
            s.clip_(-self.lr, self.lr)

        # print(f"debug: {nonzero_indices.shape[0]} nonzero gradient indices out of {g_vec.shape[0]} total parameters.")
        # s_vec = s.as_1d_tensor()
        # nonzero_s = s_vec.nonzero(as_tuple=True)[0]
        # print(f"debug: {nonzero_s.shape[0]} nonzero update indices out of {s_vec.shape[0]} total parameters.")

        return s

    def _handle_new_parameters(self, g):
        g_vec = g.as_1d_tensor()
        D_smoothed_vec = self.D_smoothed.as_1d_tensor()
        new_params_mask = (D_smoothed_vec == 0.0) & (g_vec != 0.0)
        if new_params_mask.any():
            beta2 = self.betas[1]
            # print(f"New parameters detected, initializing their D_smoothed values.")
            self.D_iter_vec[new_params_mask] = 1
            D_smoothed_vec[new_params_mask] = (1 - beta2) * g_vec[new_params_mask].abs()
            D_est_vec = self.D_est.as_1d_tensor()
            D_est_vec = D_smoothed_vec / (1 - beta2 ** self.D_iter_vec + self.eps) * (self.D_iter_vec / self.total_D_iter)
            self.D_smoothed.load_1d_tensor(D_smoothed_vec)
            self.D_est.load_1d_tensor(D_est_vec)

    def update_diagonal(self, JTJv_func, Dhat_func, z_gen_func, S):
        self.total_D_iter += 1
        beta2 = self.betas[1]

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

        D_est_t_vec = D_est_t.as_1d_tensor()
        nonzero_indices = D_est_t_vec.nonzero(as_tuple=True)[0]

        self.D_iter_vec[nonzero_indices] += 1

        D_smoothed_vec = self.D_smoothed.as_1d_tensor()
        D_smoothed_vec[nonzero_indices] = beta2 * D_smoothed_vec[nonzero_indices] + (1 - beta2) * D_est_t_vec[nonzero_indices]
        D_est_vec = self.D_est.as_1d_tensor()
        D_est_vec = D_smoothed_vec / (1 - beta2 ** self.D_iter_vec + self.eps) * (self.D_iter_vec / self.total_D_iter)
        self.D_smoothed.load_1d_tensor(D_smoothed_vec)
        self.D_est.load_1d_tensor(D_est_vec)

        # self.D_iter += 1
        # self.D_smoothed = beta2 * self.D_smoothed + (1 - beta2) * D_est_t
        # self.D_est = self.D_smoothed / (1 - beta2 ** self.D_iter)

