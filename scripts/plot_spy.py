import torch
import matplotlib.pyplot as plt

iteration = 501

J = torch.load(f"saved_output/sgd_picasso1/J_iter{iteration:05d}.pt")
H = torch.load(f"saved_output/sgd_picasso1/H_iter{iteration:05d}.pt")

# Print the sparsity of J and H
sparsity_J = 100.0 * (1.0 - (J != 0).float().mean().item())
sparsity_H = 100.0 * (1.0 - (H != 0).float().mean().item())
print(f"Sparsity of J at iteration {iteration}: {sparsity_J:.2f}%")
print(f"Sparsity of H at iteration {iteration}: {sparsity_H:.2f}%")

# Plot the spy plot of J only
plt.figure(figsize=(2, 8))
plt.spy(J.cpu(), markersize=1)

# Disable ticks
plt.xticks([])
plt.yticks([])
plt.savefig(f'figures/spy_J_iter{iteration:05d}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
# Plot the spy plot of H only
plt.figure(figsize=(8, 8))
plt.spy(H.cpu(), markersize=1)
# Disable ticks
plt.xticks([])
plt.yticks([])
# Bounding box tight
plt.savefig(f'figures/spy_H_iter{iteration:05d}.png', dpi=300, bbox_inches='tight')

