import torch

class AdamOptimizer:
    def __init__(self, lr=1.0, betas=(0.9, 0.999), eps=1e-15, clip=False):
        self.lr_init = lr
        self.betas = betas
        self.eps = eps
        self.clip = clip

        self.reset()


    def reset(self):
        self.iter = 0
        self.m = 0
        self.v = 0
        self.lr = self.lr_init

    def update_lr(self, lr):
        self.lr = lr

    def get_update(self, g):
        self.iter += 1
        bias_correction1 = 1 - self.betas[0] ** self.iter
        bias_correction2 = 1 - self.betas[1] ** self.iter

        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * g
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (g * g)

        m_hat = self.m / bias_correction1
        v_hat = self.v / bias_correction2

        s = -m_hat / (v_hat.sqrt() + self.eps)

        if self.clip:
            s.clip_(min=-1.0, max=1.0)

        s *= self.lr

        return s
