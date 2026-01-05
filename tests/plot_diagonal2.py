#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state, get_expon_lr_func, safe_interact
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
from contextlib import contextmanager
from PIL import Image

from functools import partial
from solver.gaussian_model_state import GaussianModelState, GaussianModelScaleMatrix, GaussianModelParamGroupMask, GaussianModelSplatMask
from solver.training_loss import scalar_training_loss
from solver.batch_training_loss import batch_training_loss
from solver.training_loss_hessian import scalar_training_loss_hessian
from solver.reference_training_loss import reference_training_loss
from solver.loss_image_state import MultiBatchLossImageState
from solver.solver_functions import LinearSolverFunctions
from solver.conjugate_gradient import cg_damped, cgls_damped
from solver.preconditioner import AdaHessianPreconditioner
from solver.solver_utils import CamProvider

from copy import deepcopy

from matplotlib import pyplot as plt

path_names = ["diagonal_estimate_static_no-rescaling.pth", "diagonal_estimate_static_rescaling.pth", "diagonal_estimate_dynamic_rescaling.pth"]
config_names = ["static no-rescaling", "static rescaling", "dynamic rescaling"]

for test_i in range(len(path_names)):
    path_name = path_names[test_i]
    config_name = config_names[test_i]
    f = torch.load(path_name)

    Ds = f[0]
    D_sqs = f[1]

    import code; code.interact(local=locals())

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,10))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan']

    for i, it in enumerate(Ds.keys()):
        xyz = Ds[it].xyz_grad.cpu().numpy()
        ax1.hist(xyz.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")
        ax1.axvline(xyz.min(), linestyle='dashed', linewidth=1, label=f"Iter {it} min", color=colors[i % len(colors)])

        features_dc = Ds[it].features_dc_grad.cpu().numpy()
        ax2.hist(features_dc.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")
        ax2.axvline(features_dc.min(), linestyle='dashed', linewidth=1, label=f"Iter {it} min", color=colors[i % len(colors)])

        features_rest = Ds[it].features_rest_grad.cpu().numpy()
        ax3.hist(features_rest.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")
        ax3.axvline(features_rest.min(), linestyle='dashed', linewidth=1, label=f"Iter {it} min", color=colors[i % len(colors)])

        scaling = Ds[it].scaling_grad.cpu().numpy()
        ax4.hist(scaling.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")
        ax4.axvline(scaling.min(), linestyle='dashed', linewidth=1, label=f"Iter {it} min", color=colors[i % len(colors)])

        rotation = Ds[it].rotation_grad.cpu().numpy()
        ax5.hist(rotation.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")
        ax5.axvline(rotation.min(), linestyle='dashed', linewidth=1, label=f"Iter {it} min", color=colors[i % len(colors)])

        opacity = Ds[it].opacity_grad.cpu().numpy()
        ax6.hist(opacity.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")
        ax6.axvline(opacity.min(), linestyle='dashed', linewidth=1, label=f"Iter {it} min", color=colors[i % len(colors)])

    ax1.set_yscale("log")
    ax1.set_title(f"Position")
    ax1.legend()
    ax2.set_yscale("log")
    ax2.set_title(f"Feature DC")
    ax2.legend()
    ax3.set_yscale("log")
    ax3.set_title(f"Feature Rest")
    ax3.legend()
    ax4.set_yscale("log")
    ax4.set_title(f"Scaling")
    ax4.legend()
    ax5.set_yscale("log")
    ax5.set_title(f"Rotation")
    ax5.legend()
    ax6.set_yscale("log")
    ax6.set_title(f"Opacity")
    ax6.legend()

    fig.suptitle(f"Diagonal Estimate - {config_name}")
    plt.savefig(f"figures/diagonal_estimate_{config_name.replace(' ', '_')}.png")

exit()


iters = torch.load(path_names[0])[0].keys()
figs = {}
axes = {}

for it in iters:
    figs[it], axes[it] = plt.subplots(2, 3, figsize=(15,10))

for test_i in range(len(path_names)):
    path_name = path_names[test_i]
    config_name = config_names[test_i]
    color = colors[test_i]
    f = torch.load(path_name)

    Ds = f[0]
    D_sqs = f[1]

    import code; code.interact(local=locals())

    for it in Ds.keys():
        D = Ds[it]

        xyz = D.xyz_grad.cpu().numpy()
        xyz_bins = np.linspace(-2, 20, 100)
        axes[it][0,0].hist(xyz.flatten(), bins=xyz_bins, alpha=0.5, density=True, label=f"{config_name}", color=color)
        axes[it][0,0].axvline(xyz.min(), color=color, linestyle='dashed', linewidth=1, label=f"{config_name} min")

        features_dc = D.features_dc_grad.cpu().numpy()
        features_dc_bins = np.linspace(-1, 1, 100)
        axes[it][0,1].hist(features_dc.flatten(), bins=features_dc_bins, alpha=0.5, density=True, label=f"{config_name}", color=color)
        axes[it][0,1].axvline(features_dc.min(), color=color, linestyle='dashed', linewidth=1, label=f"{config_name} min")

        features_rest = D.features_rest_grad.cpu().numpy()
        
        axes[it][0,2].hist(features_rest.flatten(), bins=100, alpha=0.5, density=True, label=f"{config_name}", color=color)
        axes[it][0,2].axvline(features_rest.min(), color=color, linestyle='dashed', linewidth=1, label=f"{config_name} min")

        scaling = D.scaling_grad.cpu().numpy()
        axes[it][1,0].hist(scaling.flatten(), bins=100, alpha=0.5, density=True, label=f"{config_name}", color=color)
        axes[it][1,0].axvline(scaling.min(), color=color, linestyle='dashed', linewidth=1, label=f"{config_name} min")

        rotation = D.rotation_grad.cpu().numpy()
        axes[it][1,1].hist(rotation.flatten(), bins=100, alpha=0.5, density=True, label=f"{config_name}", color=color)
        axes[it][1,1].axvline(rotation.min(), color=color, linestyle='dashed', linewidth=1, label=f"{config_name} min")

        opacity = D.opacity_grad.cpu().numpy()
        axes[it][1,2].hist(opacity.flatten(), bins=100, alpha=0.5, density=True, label=f"{config_name}", color=color)
        axes[it][1,2].axvline(opacity.min(), color=color, linestyle='dashed', linewidth=1, label=f"{config_name} min")


for it in iters:
    axes[it][0,0].set_yscale("log")
    axes[it][0,0].set_title(f"Iter {it} Position")
    axes[it][0,0].legend()
    axes[it][0,0].set_xlim()

    axes[it][0,1].set_yscale("log")
    axes[it][0,1].set_title(f"Iter {it} Feature DC")
    axes[it][0,1].legend()

    axes[it][0,2].set_yscale("log")
    axes[it][0,2].set_title(f"Iter {it} Feature Rest")
    axes[it][0,2].legend()

    axes[it][1,0].set_yscale("log")
    axes[it][1,0].set_title(f"Iter {it} Scaling")
    axes[it][1,0].legend()

    axes[it][1,1].set_yscale("log")
    axes[it][1,1].set_title(f"Iter {it} Rotation")
    axes[it][1,1].legend()

    axes[it][1,2].set_yscale("log")
    axes[it][1,2].set_title(f"Iter {it} Opacity")
    axes[it][1,2].legend()

    figs[it].savefig(f"figures/diagonal_estimate_iter_{it}.png")

exit()


    

f = torch.load("diagonal_estimate_static_no-rescaling.pth")

Ds_static_norescaling = f[0]
D_sqs_static_norescaling = f[1]



fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,10))

import code; code.interact(local=locals())

for it in Ds.keys():

    D = Ds[it]
    xyz = D.xyz_grad.cpu().numpy()
    ax1.hist(xyz.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")

    features_dc = D.features_dc_grad.cpu().numpy()
    ax2.hist(features_dc.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")

    features_rest = D.features_rest_grad.cpu().numpy()
    ax3.hist(features_rest.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")

    scaling = D.scaling_grad.cpu().numpy()
    ax4.hist(scaling.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")

    rotation = D.rotation_grad.cpu().numpy()
    ax5.hist(rotation.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")

    opacity = D.opacity_grad.cpu().numpy()
    ax6.hist(opacity.flatten(), bins=100, alpha=0.5, density=True, label=f"Iter {it}")


    print(xyz.shape)

# ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_title("Position")
ax1.legend()

# ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_title("Feature DC")

# ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Feature Rest")

# ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_title("Scaling")

# ax5.set_xscale("log")
ax5.set_yscale("log")
ax5.set_title("Rotation")

# ax6.set_xscale("log")
ax6.set_yscale("log")
ax6.set_title("Opacity")

plt.savefig("figures/diagonal_estimate.png")
