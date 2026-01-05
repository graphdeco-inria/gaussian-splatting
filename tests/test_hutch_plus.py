import torch
import numpy as np
import matplotlib.pyplot as plt

m = 50000
n = 10000
J = torch.randn((m, n), device="cuda")


# Add some random scaling to columns of J
c = 1.2
for i in range(n):
    J[:, i] *= ((i + 1) ** (-c)) 
print(f"s[0] = {(1) ** (-c):.6e}, s[n-1] = {(n) ** (-c):.6e}")

H = J.T @ J     # n x n
D = torch.diagonal(H)

D_ests_hutch_plus = []
D_ests_hutch = []
D_sq_ests_hutch = []

ds = [10, 50, 100, 200, 500]

errors_hutch_plus = {}
errors_hutch = {}
errors_sq_hutch = {}

for d in ds:

    # Random +-1 tensor of (2, n, d)
    SG = (2 * torch.randint(0, 2, (2, n, d), device="cuda") - 1).float()

    S = SG[0]
    G = SG[1]

    HS = H @ S      # n x d

    # Compute the QR of HS
    Q, R = torch.linalg.qr(HS)      # Q is n x d, R is d x d


    QQTHQQT = Q @ (Q.T @ H @ Q) @ Q.T    # n x n
    D_est1 = torch.diagonal(QQTHQQT)     # n,

    del QQTHQQT

    I_QQT = torch.eye(n, device="cuda") - Q @ Q.T    # n x n
    I_QQT_H_I_QQT = I_QQT @ H @ I_QQT    # n x n

    D_est2 = torch.sum(G * (I_QQT_H_I_QQT @ G), dim=1) / d    # n,

    D_est = D_est1 + D_est2

    D_ests_hutch_plus.append(D_est.cpu().numpy())

    error = torch.norm(D - D_est) / torch.norm(D)

    errors_hutch_plus[d * 3] = error.item()

    print(f"Probes: {d*3}, Hutch++ Error: {errors_hutch_plus[d*3]:.6f}")

for d in ds:

    # Random +-1 tensor of (n, 3*d)
    W = (2 * torch.randint(0, 2, (n, d * 3), device="cuda") - 1).float()

    HW = H @ W      # n x (3*d)
    D_est = torch.sum(W * HW, dim=1) / (d * 3)

    D_ests_hutch.append(D_est.cpu().numpy())

    error = torch.norm(D - D_est) / torch.norm(D)
    errors_hutch[d * 3] = error.item()

    print(f"Probes: {d*3}, Hutchinson Error: {errors_hutch[d*3]:.6f}, Hutch++ Error: {errors_hutch_plus[d*3]:.6f}")

for d in ds:
    # Random +-1 tensor of (n, 3*d)
    W = (2 * torch.randint(0, 2, (n, d * 3), device="cuda") - 1).float()

    HW = H @ W      # n x (3*d)
    D_ests = W * HW
    D_sq_ests = D_ests * D_ests
    D_sq = torch.sum(D_sq_ests, dim=1) / (d * 3)

    D_est = torch.sqrt(D_sq)

    D_sq_ests_hutch.append(D_est.cpu().numpy())
    error = torch.norm(D - D_est) / torch.norm(D)
    errors_sq_hutch[d * 3] = error.item()

plt.figure(figsize=(8,6))
plt.plot(list(errors_hutch.keys()), list(errors_hutch.values()), marker='o', label='Hutchinson')
plt.plot(list(errors_hutch_plus.keys()), list(errors_hutch_plus.values()), marker='o', label='Hutch++')

plt.xlabel('Number of Probes')
plt.ylabel('Relative Error in Diagonal Estimate')
plt.title('Hutchinson vs Hutch++ Diagonal Estimation Error')
plt.legend()

plt.savefig('figures/hutchinson_vs_hutch_plus.png')


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))

for i, d in enumerate(ds):
    ax1.plot(np.arange(n), D_ests_hutch[i], alpha=0.5, label=f'Num iters {d*3}')
    ax2.plot(np.arange(n), D_ests_hutch_plus[i], alpha=0.5, label=f'Num iters {d*3}')
    ax3.plot(np.arange(n), D_sq_ests_hutch[i], alpha=0.5, label=f'Num iters {d*3}')

ax1.plot(np.arange(n), D.abs().cpu().numpy(), label='D', color='black', linewidth=1)
ax2.plot(np.arange(n), D.abs().cpu().numpy(), label='D', color='black', linewidth=1)
ax3.plot(np.arange(n), D.cpu().numpy(), label='D', color='black', linewidth=1)

ax1.set_title('Hutchinson Diagonal Estimates')
ax1.legend()
ax1.set_yscale('log')

ax2.set_title('Hutch++ Diagonal Estimates')
ax2.legend()
ax2.set_yscale('log')

ax3.set_title('Hutchinson Square Root Diagonal Estimates')
ax3.legend()
ax3.set_yscale('log')

fig.savefig('figures/diagonal_estimates_comparison.png')


