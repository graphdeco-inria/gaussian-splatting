import torch
import matplotlib.pyplot as plt

def restarted_hutchinsons_method(H, num_iter, restart_iter=-1, 
                                 D_init=torch.tensor(1.0, device="cuda"), debug=False):
    n = H.shape[0]

    D_accum = 0

    S = (1 / (D_init + 1e-15)).sqrt()

    accum_iter = 0

    eps = 1e-15

    for i in range(num_iter):
        accum_iter += 1
        z = torch.randint(0, 2, (n, ), device="cuda").float() * 2.0 - 1.0
        Sz = S * z
        HSz = H @ Sz
        D_est_i = (Sz * HSz) / (S * S)


        D_accum += D_est_i

        if (restart_iter != -1 and accum_iter % restart_iter == 0) or (i + 1 == num_iter):
            print("we are here")
            D_est = D_accum.abs() / accum_iter

            S = (1 / (D_est + 1e-15)).sqrt()

            D_accum = 0
            accum_iter = 0

            if debug:
                import code; code.interact(local=locals(), banner="debug diag est")


    return D_est

H1, D1_est_debug = torch.load("debug_malicious_gaussian_H_removed.pth")
H2, D2_est_debug = torch.load("debug_malicious_gaussian_H.pth")

D_init = torch.load("debug_malicious_gaussian_D_init.pth")

n = H1.shape[0]

D1 = H1.diagonal()
D2 = H2.diagonal()

D1_est = restarted_hutchinsons_method(H1, 20, -1,) # D_init=D_init)
D2_est = restarted_hutchinsons_method(H2, 20, -1, debug=True) # D_init=D_init)

bad_mask = (D1 == 0.0) & (D2 != 0.0)

figure, ax = plt.subplots(figsize=(8, 6))

sorted_indices = torch.argsort(D2, descending=True)

# plt.plot(range(n), D_init[sorted_indices].cpu().numpy(), label="Initial Diagonal", color='black')
# plt.plot(range(n), D1_est_debug[sorted_indices].cpu().numpy(), label="Saved Diagonal H1 (removed)", color='cyan')
# plt.plot(range(n), D2_est_debug[sorted_indices].cpu().numpy(), label="Saved Diagonal H2 (malicious)", color='orange')
plt.plot(range(n), D1_est[sorted_indices].cpu().numpy(), label="Estimated Diagonal H1 (removed)", color='green')
plt.plot(range(n), D2_est[sorted_indices].cpu().numpy(), label="Estimated Diagonal H2 (malicious)", color='magenta')
plt.plot(range(n), D1[sorted_indices].cpu().numpy(), label="True Diagonal H1 (removed)", color='blue')
plt.plot(range(n), D2[sorted_indices].cpu().numpy(), label="True Diagonal H2 (malicious)", color='red')

plt.legend()
plt.yscale('log')
ax.set_xlim(0, 4500)

plt.savefig("figures/debug_malicious_gaussian_diagonal_comparison.png")



l1_norms = H2.abs().sum(dim=1)
linf_norms = H2.abs().max(dim=1).values

figure, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(n), l1_norms[sorted_indices].cpu().numpy(), label="L1", color='green')
plt.plot(range(n), linf_norms[sorted_indices].cpu().numpy(), label="L_inf", color='blue')
plt.legend()
plt.yscale('log')
ax.set_xlim(0, 4500)
plt.savefig("figures/debug_malicious_gaussian_H_error.png")

nonzeros1 = H1.count_nonzero(dim=1)
nonzeros2 = H2.count_nonzero(dim=1)

figure, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(n), nonzeros2[sorted_indices].cpu().numpy(), label="Non-zeros H2", color='red')
plt.plot(range(n), nonzeros1[sorted_indices].cpu().numpy(), label="Non-zeros H1", color='blue')
plt.legend()
ax.set_xlim(0, 4500)
plt.savefig("figures/debug_malicious_gaussian_nonzero.png")

import code; code.interact(local=locals())
