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


Ds = torch.load("diagonal_abs_estimate.pth")


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,10))

for it in [10, 100, 500, 1000, 10000, 20000]:
    if it not in Ds:
        continue

    D = Ds[it]
    xyz = D.xyz_grad.cpu().numpy()
    xyz_logbins = np.logspace(np.log10(xyz[xyz > 0.0].min()), np.log10(xyz.max()), num=100)
    ax1.hist(xyz.flatten(), bins=xyz_logbins, alpha=0.5, density=True, label=f"Iter {it}")

    features_dc = D.features_dc_grad.cpu().numpy()
    features_dc_logbins = np.logspace(np.log10(features_dc[features_dc > 0.0].min()), np.log10(features_dc.max()), num=100)
    ax2.hist(features_dc.flatten(), bins=features_dc_logbins, alpha=0.5, density=True, label=f"Iter {it}")

    features_rest = D.features_rest_grad.cpu().numpy()
    features_rest_logbins = np.logspace(np.log10(features_rest[features_rest > 0.0].min()), np.log10(features_rest.max()), num=100)
    ax3.hist(features_rest.flatten(), bins=features_rest_logbins, alpha=0.5, density=True, label=f"Iter {it}")
    scaling = D.scaling_grad.cpu().numpy()

    scaling = D.scaling_grad.cpu().numpy()
    scaling_logbins = np.logspace(np.log10(scaling[scaling > 0.0].min()), np.log10(scaling.max()), num=100)
    ax4.hist(scaling.flatten(), bins=scaling_logbins, alpha=0.5, density=True, label=f"Iter {it}")

    rotation = D.rotation_grad.cpu().numpy()
    rotation_logbins = np.logspace(np.log10(rotation[rotation > 0.0].min()), np.log10(rotation.max()), num=100)
    ax5.hist(rotation.flatten(), bins=rotation_logbins, alpha=0.5, density=True, label=f"Iter {it}")

    opacity = D.opacity_grad.cpu().numpy()
    opacity_logbins = np.logspace(np.log10(opacity[opacity > 0.0].min()), np.log10(opacity.max()), num=100)
    ax6.hist(opacity.flatten(), bins=opacity_logbins, alpha=0.5, density=True, label=f"Iter {it}")



    print(xyz.shape)

# ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.set_title("Position")
ax1.legend()

# ax2.set_xscale("log")
# ax2.set_yscale("log")
ax2.set_title("Feature DC")

# ax3.set_xscale("log")
# ax3.set_yscale("log")
ax3.set_title("Feature Rest")

# ax4.set_xscale("log")
# ax4.set_yscale("log")
ax4.set_title("Scaling")

# ax5.set_xscale("log")
# ax5.set_yscale("log")
ax5.set_title("Rotation")

# ax6.set_xscale("log")
# ax6.set_yscale("log")
ax6.set_title("Opacity")

plt.savefig("figures/diagonal_estimate.png")
