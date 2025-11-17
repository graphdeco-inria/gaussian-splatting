import torch
import numpy as np
import matplotlib.pyplot as plt

def apply_preconditioner(H, D, damp=0.0):
    D_inv_sqrt = 1.0 / (D + damp).sqrt()
    H_precond = torch.diag(D_inv_sqrt) @ apply_damp(H, damp) @ torch.diag(D_inv_sqrt)
    return H_precond

def apply_damp(H, damp):
    n = H.shape[0]
    H_damped = H + damp * torch.eye(n, device=H.device)
    return H_damped

def apply_damp_to_diag(Ds, damp):
    for it in Ds.keys():
        Ds[it] += damp

def compute_conds(H, Ds, damp=0.0):
    conds = {}
    for it in Ds.keys():
        D_est = Ds[it]
        cond = torch.linalg.cond(apply_preconditioner(H, D_est, damp))
        conds[it] = cond.item()

    return conds

class H_block:
    def __init__(self, m, p, num_param_groups, param_group_scales=None, param_scales=None, num_param_sel=0.3):
        self.Ji = torch.randn((m, num_param_groups, p), device="cuda")

        # Apply a parameter group scaling
        if param_group_scales is not None:
            self.Ji *= param_group_scales.view(1, num_param_groups, 1).cuda()

        # Apply a per parameter scaling
        if param_scales is not None:
            self.Ji *= param_scales.view(1, 1, p).cuda()

        # Apply a per parameter mask
        indices = torch.randperm(p)
        percentages = torch.arange(0, p) / p
        param_scales = 1.0 / (1.0 + torch.exp(20 * (percentages - num_param_sel))) * torch.exp(-5 * percentages)
        # print("param_scales = ", param_scales)
        param_scales = param_scales[indices].to(self.Ji.device).view(1, 1, p)

        self.Ji = self.Ji * param_scales

        self.Hi = self.Ji.view(m, -1).T @ self.Ji.view(m, -1)  # (num_param_groups*n) x (num_param_groups*n)

        self.Ji = self.Ji.cpu()
        self.Hi = self.Hi.cpu()

        self.e = 0.0

    def update_noise(self, noise=0.0):
        self.e = torch.randn(self.Ji.shape, device="cuda") * noise

    def H(self, with_noise=False):
        if not with_noise:
            return self.Hi.cuda()
        else:
            Ji = self.Ji.cuda()
            Ji *= (1.0 + self.e)
            m, num_param_groups, p = Ji.shape
            Hi = Ji.view(m, -1).T @ Ji.view(m, -1)
            return self.Hi.cuda()

def init_H_blocks(num_blocks, m, p, num_param_groups, param_group_scales=None, param_scales=None, num_param_sel=0.3, noise=0.01):
    H_blocks = []
    for i in range(num_blocks):
        print(f"Initializing H block {i+1}/{num_blocks}")
        H_blocks.append(H_block(m, p, num_param_groups, param_group_scales, param_scales, num_param_sel))
        H_blocks[-1].update_noise(noise)
    return H_blocks

def hutchinsons(H_blocks, S_hat, num_probes=[], samples_per_probe=1):
    M = len(H_blocks)
    H_hat = torch.zeros_like(H_blocks[0].H())
    n = H_hat.shape[0]

    D_accum = torch.zeros((n, ), device="cuda")
    
    num_probes = sorted(num_probes)
    for num_iters in num_probes:
        assert num_iters % samples_per_probe == 0

    max_num_probes = max(num_probes)

    D_ests = {}

    for it in range(0, max_num_probes, samples_per_probe):
        sampled_indices = np.random.choice(M, size=samples_per_probe, replace=False)
        scale = M / samples_per_probe

        w = torch.randint(0, 2, (n, ), device="cuda") * 2 - 1  # Random +-1 vector (n, )
        Sw = S_hat * w

        Hi = torch.zeros_like(H_hat)

        for i in sampled_indices:
            Hi += H_blocks[i].H(with_noise=True)

        Hi *= scale

        Di = (Sw * (Hi @ Sw)) / (S_hat * S_hat)

        D_accum += Di

        if (it + samples_per_probe) in num_probes:
            D_hat = D_accum / (it + samples_per_probe)
            D_ests[it + samples_per_probe] = D_hat
            print("save", it + samples_per_probe)

    return D_ests

def squared_hutchinsons(H_blocks, S_hat, num_probes=[], samples_per_probe=1, restart_iters=-1):
    M = len(H_blocks)
    H_hat = torch.zeros_like(H_blocks[0].H())
    n = H_hat.shape[0]

    D_sq_accum = torch.zeros((n, ), device="cuda")
    denom_iters = 0
    
    num_probes = sorted(num_probes)
    for num_iters in num_probes:
        assert num_iters % samples_per_probe == 0

    if restart_iters > 0:
        assert restart_iters % samples_per_probe == 0

    max_num_probes = max(num_probes)

    D_ests = {}

    for it in range(0, max_num_probes, samples_per_probe):
        denom_iters += samples_per_probe

        sampled_indices = np.random.choice(M, size=samples_per_probe, replace=False)
        scale = M / samples_per_probe

        w = torch.randint(0, 2, (n, ), device="cuda") * 2 - 1  # Random +-1 vector (n, )
        Sw = S_hat * w

        Hi = torch.zeros_like(H_hat)

        for i in sampled_indices:
            Hi += H_blocks[i].H(with_noise=True)

        Hi *= scale

        Di = (Sw * (Hi @ Sw)) / (S_hat * S_hat)

        D_sq_accum += Di * Di

        if (it + samples_per_probe) in num_probes:
            D_sq_hat = D_sq_accum / denom_iters
            D_est = D_sq_hat.sqrt()
            D_ests[it + samples_per_probe] = D_est
            print("save", it + samples_per_probe)

        if restart_iters > 0 and (it + samples_per_probe) % restart_iters == 0:
            D_sq_hat = D_sq_accum / denom_iters
            D_est = D_sq_hat.sqrt()
            S_hat = 1 / D_est.sqrt()
            D_sq_accum *= 0
            denom_iters = 0
            print("restart at ", it + samples_per_probe)

    return D_ests

def main():
    num_images = 200
    m = 5000
    n = 2000
    num_param_groups = 5
    p = n // num_param_groups
    num_param_sel = 0.1
    param_group_scales = torch.tensor([1e0, 1e-1, 1e-2, 1e-3, 1e-4], device="cuda")
    param_scales = 1.0 / ((torch.arange(p, device="cuda") + 1) / p) * 1e-4
    S_hat_noise = 1e2
    J_noise = 0.0 # 1.0
    damp = 1e-12
    # num_probes = [5, 10, 20, 50, 100, 1000]
    num_probes = [10, 20, 50, 100]
    samples_per_probe = 1
    restart_iters = -1

    H_blocks = init_H_blocks(num_blocks=num_images, m=m, p=p, 
                             num_param_groups=num_param_groups, 
                             param_group_scales=param_group_scales, 
                             param_scales=param_scales,
                             num_param_sel=num_param_sel,
                             noise=J_noise)

    H = torch.zeros((n, n), device="cuda")
    H_hat = torch.zeros((n, n), device="cuda")

    # Aggregate H
    for Hi in H_blocks:
        H += Hi.H(with_noise=False)
        H_hat += Hi.H(with_noise=True)

    D = torch.diagonal(H) + damp
    D_hat = torch.diagonal(H_hat) + damp

    sorted_indices = torch.argsort(D.view(num_param_groups, p), descending=True, dim=1)
    offsets = (torch.arange(num_param_groups, device="cuda") * p).unsqueeze(-1)
    sorted_indices = (sorted_indices + offsets).flatten()

    H_hat_cond = torch.linalg.cond(apply_damp(H_hat, damp))
    DHD_hat_cond = torch.linalg.cond(apply_preconditioner(H_hat, D_hat, damp))
    print("H_hat_cond = ", H_hat_cond.item())
    print("DHD_hat_cond = ", DHD_hat_cond.item())


    ones = torch.ones((n, ), device="cuda")
    S_hat_inv = (param_group_scales * (S_hat_noise ** (torch.rand(param_group_scales.shape, device="cuda") * 2 - 1))).sqrt()
    print("S_hat_inv = ", S_hat_inv)
    S_hat = (torch.ones((num_param_groups, n // num_param_groups), device="cuda") / S_hat_inv.view(num_param_groups, 1)).view(n, )

    ################# Hutchinson Square Root without Rescaling ######################### 
    D_ests_hutch_sq = squared_hutchinsons(H_blocks, ones, num_probes=num_probes, samples_per_probe=samples_per_probe, restart_iters=restart_iters)
    conds = compute_conds(H_hat, D_ests_hutch_sq, damp)
    print("conds = ", conds)
    S_hat2 = 1 / D_ests_hutch_sq[num_probes[-1]].sqrt()

    ################# Hutchinson Square Root with Rescaling ######################### 
    D_ests_hutch_sq_rescaling = squared_hutchinsons(H_blocks, S_hat, num_probes=num_probes, samples_per_probe=samples_per_probe)
    conds = compute_conds(H_hat, D_ests_hutch_sq_rescaling, damp)
    print("conds = ", conds)

    ################# Hutchinson with Rescaling ######################### 
    D_ests_hutch = hutchinsons(H_blocks, ones, num_probes=num_probes, samples_per_probe=samples_per_probe)

    ################# Hutchinson with Rescaling ######################### 
    D_ests_hutch_rescaling = hutchinsons(H_blocks, S_hat, num_probes=num_probes, samples_per_probe=samples_per_probe)

    ################# Update noise ##################################
    H_hat *= 0.0
    for Hi in H_blocks:
        Hi.update_noise(J_noise)
        H_hat += Hi.H(with_noise=True)

    ################# Hutchinson with Reused Rescaling ######################### 
    D_ests_hutch_sq_rescaling2 = squared_hutchinsons(H_blocks, S_hat2, num_probes=num_probes, samples_per_probe=samples_per_probe)
    conds = compute_conds(H_hat, D_ests_hutch_sq_rescaling2, damp)
    print("conds = ", conds)
    S_hat3 = 1 / D_ests_hutch_sq_rescaling2[num_probes[-1]].sqrt()

    ################# Update noise ##################################
    H_hat *= 0.0
    for Hi in H_blocks:
        Hi.update_noise(J_noise)
        H_hat += Hi.H(with_noise=True)

    ################# Hutchinson with Reused Rescaling2 ######################### 
    D_ests_hutch_sq_rescaling3 = squared_hutchinsons(H_blocks, S_hat3, num_probes=num_probes, samples_per_probe=samples_per_probe)
    conds = compute_conds(H_hat, D_ests_hutch_sq_rescaling3, damp)
    print("conds = ", conds)
    S_hat4 = 1 / D_ests_hutch_sq_rescaling3[num_probes[-1]].sqrt()

    ################# Hutchinson with Reused Rescaling3 ######################### 
    D_ests_hutch_sq_rescaling4 = squared_hutchinsons(H_blocks, S_hat4, num_probes=num_probes, samples_per_probe=samples_per_probe)
    conds = compute_conds(H_hat, D_ests_hutch_sq_rescaling4, damp)
    print("conds = ", conds)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(16,20))

    for it in D_ests_hutch_sq.keys():
        ax1.plot(np.arange(n), D_ests_hutch_sq[it][sorted_indices].cpu().numpy(), alpha=0.5, label=f'Num iters {it}')
    for it in D_ests_hutch_sq_rescaling.keys():
        ax2.plot(np.arange(n), D_ests_hutch_sq_rescaling[it][sorted_indices].cpu().numpy(), alpha=0.5, label=f'Num iters {it}')
    for it in D_ests_hutch.keys():
        ax3.plot(np.arange(n), D_ests_hutch[it][sorted_indices].cpu().numpy(), alpha=0.5, label=f'Num iters {it}')
    for it in D_ests_hutch_rescaling.keys():
        ax4.plot(np.arange(n), D_ests_hutch_rescaling[it][sorted_indices].cpu().numpy(), alpha=0.5, label=f'Num iters {it}')
    for it in D_ests_hutch_sq_rescaling2.keys():
        ax5.plot(np.arange(n), D_ests_hutch_sq_rescaling2[it][sorted_indices].cpu().numpy(), alpha=0.5, label=f'Num iters {it}')
    for it in D_ests_hutch_sq_rescaling3.keys():
        ax6.plot(np.arange(n), D_ests_hutch_sq_rescaling3[it][sorted_indices].cpu().numpy(), alpha=0.5, label=f'Num iters {it}')
    for it in D_ests_hutch_sq_rescaling4.keys():
        ax7.plot(np.arange(n), D_ests_hutch_sq_rescaling4[it][sorted_indices].cpu().numpy(), alpha=0.5, label=f'Num iters {it}')

    ax1.plot(np.arange(n), D_hat[sorted_indices].cpu().numpy(), color="purple", label="D_hat")
    ax1.plot(np.arange(n), D[sorted_indices].cpu().numpy(), color="black", label="D")
    ax2.plot(np.arange(n), D_hat[sorted_indices].cpu().numpy(), color="purple", label="D_hat")
    ax2.plot(np.arange(n), D[sorted_indices].cpu().numpy(), color="black", label="D")
    ax3.plot(np.arange(n), D_hat[sorted_indices].cpu().numpy(), color="purple", label="D_hat")
    ax3.plot(np.arange(n), D[sorted_indices].cpu().numpy(), color="black", label="D")
    ax4.plot(np.arange(n), D_hat[sorted_indices].cpu().numpy(), color="purple", label="D_hat")
    ax4.plot(np.arange(n), D[sorted_indices].cpu().numpy(), color="black", label="D")
    ax5.plot(np.arange(n), D_hat[sorted_indices].cpu().numpy(), color="purple", label="D_hat")
    ax5.plot(np.arange(n), D[sorted_indices].cpu().numpy(), color="black", label="D")
    ax6.plot(np.arange(n), D_hat[sorted_indices].cpu().numpy(), color="purple", label="D_hat")
    ax6.plot(np.arange(n), D[sorted_indices].cpu().numpy(), color="black", label="D")
    ax7.plot(np.arange(n), D_hat[sorted_indices].cpu().numpy(), color="purple", label="D_hat")
    ax7.plot(np.arange(n), D[sorted_indices].cpu().numpy(), color="black", label="D")

    ylim_max = 1e2
    ylim_min = 1e-17

    ax1.set_title('Hutchinson Sqrt Diagonal Estimates')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylim(ylim_min, ylim_max)

    ax2.set_title('Hutchinson Sqrt Diagonal Estimates with Rescaling')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_ylim(ylim_min, ylim_max)

    ax3.set_title('Hutchinson Diagonal Estimates')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.set_ylim(ylim_min, ylim_max)

    ax4.set_title('Hutchinson Diagonal Estimates without Rescaling')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.set_ylim(ylim_min, ylim_max)

    ax5.set_title('Hutchinson Sqrt Diagonal Estimates with Reused Rescaling')
    ax5.legend()
    ax5.set_yscale('log')
    ax5.set_ylim(ylim_min, ylim_max)

    ax6.set_title('Hutchinson Sqrt Diagonal Estimates with Reused Rescaling2')
    ax6.legend()
    ax6.set_yscale('log')
    ax6.set_ylim(ylim_min, ylim_max)

    ax7.set_title('Hutchinson Sqrt Diagonal Estimates with Reused Rescaling3')
    ax7.legend()
    ax7.set_yscale('log')
    ax7.set_ylim(ylim_min, ylim_max)

    fig.savefig('figures/toy_diagonal_estimates.png')

    # Run squared Hutchinson's method that we have been doing


if __name__ == "__main__":

    torch.random.manual_seed(0)
    np.random.seed(0)

    main()



