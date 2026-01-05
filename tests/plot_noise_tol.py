import matplotlib.pyplot as plt
import numpy as np
import torch
import math

iterations = [15000, 15010, 15020, 20000, 20010, 20020]

max_vals = {}
min_vals = {}

figs = {}
axeses = {}

for iteration in iterations:

    d = torch.load(f"noise_tol_losses_iter{iteration}.pth")
    noisy_losses = d["noisy_losses"]
    ref_losses = d["ref_losses"]

    image_indices = sorted(ref_losses.keys())
    param_groups = list(noisy_losses.keys())


    print(image_indices)

    num_param_groups = len(param_groups)
    num_noise_levels = len(noisy_losses[param_groups[0]].keys())

    fig, axes = plt.subplots(num_noise_levels, num_param_groups, figsize=(5 * num_param_groups, 4 * num_noise_levels), squeeze=False)

    figs[iteration] = fig
    axeses[iteration] = axes

    for ax_j, param_group in enumerate(param_groups):
        if param_group not in max_vals.keys():
            max_vals[param_group] = {}
            min_vals[param_group] = {}

        for ax_i, noise_level in enumerate(sorted(noisy_losses[param_group].keys())):
            noise_level_rounded = 10 ** int(math.log10(noise_level))

            if noise_level_rounded not in max_vals[param_group].keys():
                max_vals[param_group][noise_level_rounded] = -np.inf
                min_vals[param_group][noise_level_rounded] = np.inf

            ax = axes[ax_i, ax_j] if num_noise_levels > 1 and num_param_groups > 1 else \
                 axes[ax_i] if num_noise_levels > 1 else \
                 axes[ax_j]

            for img_i, img_idx in enumerate(image_indices):
                noisy_loss = np.array(noisy_losses[param_group][noise_level][img_idx])
                errors = noisy_loss - ref_losses[img_idx]

                # Plot scatter plot with x=img_i, y=errors
                ax.plot(img_i * np.ones_like(errors), errors, 'o')

                if max_vals[param_group][noise_level_rounded] < np.max(errors):
                    max_vals[param_group][noise_level_rounded] = np.max(errors)
                if min_vals[param_group][noise_level_rounded] > np.min(errors):
                    min_vals[param_group][noise_level_rounded] = np.min(errors)

            if ax_i == 0:
                ax.set_title(f"Param Group: {param_group}")
            if ax_j == 0:
                ax.set_ylabel(f"Noise Level: {noise_level_rounded}\nLoss Error (Noisy - Reference)")
            ax.set_xlabel("Image Index")

    fig.suptitle(f"Loss Errors vs Image Index for Different Noise Levels at Iteration {iteration}", fontsize=16)


print("min_vals:", min_vals)
print("max_vals:", max_vals)


for iteration in iterations:
    fig = figs[iteration]
    axes = axeses[iteration]

    param_groups = list(max_vals.keys())

    for ax_j, param_group in enumerate(param_groups):
        noise_levels = list(max_vals[param_group].keys())
        for ax_i, noise_level in enumerate(noise_levels):
            ax = axes[ax_i, ax_j] if len(noise_levels) > 1 and len(param_groups) > 1 else \
                 axes[ax_i] if len(noise_levels) > 1 else \
                 axes[ax_j]

            ylim_max = max_vals[param_group][noise_level] + 0.1 * abs(max_vals[param_group][noise_level])
            ylim_min = min_vals[param_group][noise_level] - 0.1 * abs(max_vals[param_group][noise_level])

            print(f"Setting ylim for param_group={param_group}, noise_level={noise_level}: ({ylim_min}, {ylim_max})")
            ax.set_ylim(ylim_min, ylim_max)

    fig.savefig(f"figures/noise_tol_loss_errors_iter{iteration}.png")



