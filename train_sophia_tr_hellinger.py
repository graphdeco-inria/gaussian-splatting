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
from solver.gaussian_model_vector import GaussianModelVector
from solver.training_loss import scalar_training_loss
from solver.batch_training_loss import batch_training_loss
from solver.training_loss_hessian import scalar_training_loss_hessian
from solver.reference_training_loss import reference_training_loss
from solver.conjugate_gradient import conjugate_gradient
from solver.diagonal_estimator import restarted_squared_hutchinson, restarted_hutchinson
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy, construct_Dhat_func
from solver.adam_optimizer import AdamOptimizer
from solver.sophia_optimizer import SophiaOptimizer
from solver.hellinger_clip import clip_hellinger, debug_hellinger

from copy import deepcopy

from matplotlib import pyplot as plt

from utils.gif_renderer import GifRenderer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Debug disabling FusedSSIM")
    FUSED_SSIM_AVAILABLE = False
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

@contextmanager
def temp_seed(seed):
    """
    Context manager to temporarily set a seed for reproducibility.
    """
    np_state = np.random.get_state()
    np.random.seed(seed)
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class ExponentialLRScheduler:
    def __init__(self, init_lr, final_lr, max_iter):
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.max_iter = max_iter

        # compute decay factor γ such that:
        # final_lr = init_lr * γ^(max_iter)
        self.gamma = (final_lr / init_lr) ** (1.0 / max_iter)

    def get_lr(self, current_iter):
        # Clamp to [0, max_iter]
        current_iter = max(0, min(current_iter, self.max_iter))
        return self.init_lr * (self.gamma ** current_iter)


class CamProvider:
    def __init__(self, cameras, scale1=1.0):
        self.cameras = cameras
        self.scale1 = scale1

    def sample_new(self, batch_size):
        indices = np.random.choice(len(self.cameras), batch_size, replace=True)
        viewpoint_batch = [self.cameras[idx] for idx in indices]
        return viewpoint_batch, self.scale1 * len(self.cameras) / batch_size

def JTJv_hat(v, JTJv_func, gaussians, cam_provider, batch_size, S=None, damp=0.0):
    viewpoint_batch, scale = cam_provider.sample_new(batch_size=batch_size)
    return JTJv_func(v=v, gaussians=gaussians, viewpoint_cams=viewpoint_batch, scale=scale, S=S, damp=damp)

def compute_ref_loss(ref_loss_func, gaussians, viewpoint_cams, scale):
    ref_loss = 0.0
    for vc_i, vc in enumerate(viewpoint_cams):
        ref_loss_i = ref_loss_func(gaussians=gaussians, viewpoint_cam=vc) ** 2
        ref_loss_i *= scale
        ref_loss += ref_loss_i.item()
    return ref_loss

def compute_JTJ_col(JTJv_func, index, gaussians):
    v = GaussianModelVector.zeros_like(gaussians)
    v_vec = v.as_1d_tensor()
    v_vec[index] = 1.0
    v.load_1d_tensor(v_vec)
    return JTJv_func(v=v).as_1d_tensor()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, gif_interval):
    splat_mask = None

    ####### Some fixed parameters #########
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    model_path = ""
    ####### Some fixed parameters #########

    testing_iterations = testing_iterations + list(range(0, opt.iterations + 1, opt.eval_interval))

    pixel_sample_rate = opt.pixel_sample_rate_max
    xyz_lr = opt.xyz_lr_init
    damp = opt.damp_init

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        checkpoint_data = torch.load(checkpoint)
        if len(checkpoint_data) == 3:
            (model_params, first_iter, D_est) = checkpoint_data
        else:
            (model_params, first_iter) = checkpoint_data
        gaussians.restore(model_params, opt)
        del checkpoint_data

    xyz_lr_scheduler = ExponentialLRScheduler(opt.xyz_lr_init, opt.xyz_lr_final, opt.xyz_lr_max_steps)
    xyz_lr = xyz_lr_scheduler.get_lr(first_iter)

    lr = GaussianModelVector(xyz=xyz_lr,  
                             features_dc=opt.features_dc_lr, 
                             features_rest=opt.features_rest_lr,
                             scaling=opt.scaling_lr,
                             rotation=opt.rotation_lr,
                             opacity=opt.opacity_lr,
                             exposure=opt.exposure_lr,
                             gaussians=gaussians)
    S = GaussianModelVector(xyz=1.0,
                            features_dc=1.0,
                            features_rest=1.0,
                            scaling=1.0,
                            rotation=1.0,
                            opacity=1.0,
                            exposure=1.0,
                            gaussians=gaussians)

    sophia_optimizer = SophiaOptimizer(lr=lr, 
                                       betas=(opt.adahessian_beta1, opt.adahessian_beta2),
                                       eps=1e-15, clip=False,
                                       gamma=opt.sophia_gamma,
                                       diagonal_update_interval=opt.diagonal_update_interval,
                                       num_init_iter=opt.diagonal_init_iter,
                                       num_init_restart_iter=opt.diagonal_init_restart_iter,
                                       num_update_iter=opt.diagonal_update_iter,
                                       num_update_restart_iter=opt.diagonal_update_restart_iter
                                       )

    sophia_losses = [[], [], [], []]
    sophia_images = [[], [], [], []]

    sophia_optimizer.reset()
    sophia_optimizer.set_clip(False)

    adam_optimizer = AdamOptimizer(lr=lr, betas=(opt.adam_beta1, opt.adam_beta2), eps=1e-15, clip=False)
    adam_optimizer.reset()

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    gradient_start = torch.cuda.Event(enable_timing = True)
    sophia_update_start = torch.cuda.Event(enable_timing = True)
    clip_start = torch.cuda.Event(enable_timing = True)
    adam_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    post_step_start = torch.cuda.Event(enable_timing = True)
    post_step_end = torch.cuda.Event(enable_timing = True)


    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    test_stack = scene.getTestCameras().copy()

    # DEBUG
    # print("Reducing viewpoints")
    # # viewpoint_stack = [viewpoint_stack[i] for i in range(150, 190, 1)]
    # test_stack = test_stack
    # safe_interact(local=locals(), banner="Debug prompt after reducing viewpoints")

    viewpoint_indices = list(range(len(viewpoint_stack)))

    preconditioner = None

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    iter_time_accum = 0.0
    gradient_time_accum = 0.0
    preconditioner_time_accum = 0.0
    update_time_accum = 0.0

    if opt.use_adam:
        safe_interact(local=locals(), banner="Using Adam optimizer - not Sophia")

    with torch.no_grad():
        training_report(tb_writer, first_iter, 0.0, 0.0, l1_loss, 0.0, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, val_indices=None, test_stack=test_stack, train_stack=viewpoint_stack)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, image_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        with torch.no_grad():
            gaussians.update_learning_rate(iteration)

            num_batch_cameras = len(viewpoint_indices) if opt.num_images == -1 else min(opt.num_images, len(viewpoint_indices))
            rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
            scale = len(viewpoint_indices) / num_batch_cameras

            viewpoint_batch = []
            for rand_idx in rand_indices:
                viewpoint_cam = viewpoint_stack[rand_idx]
                viewpoint_batch.append(viewpoint_cam)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background


            render_args = {"iteration": iteration,
                           "opt": opt,
                           "pipe": pipe,
                           "bg": bg,
                           "train_test_exp": train_test_exp,
                           "depth_l1_weight": depth_l1_weight,
                           "loss_type": opt.loss_type,
                           "huber_delta": opt.huber_delta,
                           "disable_ssim": opt.disable_ssim,
                           "batch_size": 1,
                           "pixel_mask": None,
                           }

            loss_func = construct_loss_func(**render_args)
            g_func = construct_g_func(**render_args)
            JTJv_func = construct_JTJv_func(**render_args)
            Dhat_func = construct_Dhat_func(**render_args)
            z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians)


            JTJv_func1 = partial(JTJv_func, gaussians=gaussians, viewpoint_cams=viewpoint_batch, S=S, scale=1)
            Dhat_func1 = partial(Dhat_func, gaussians=gaussians, viewpoint_cams=viewpoint_batch)

            gradient_start.record()
            loss_sophia_tr, g, batch_stats = g_func(gaussians=gaussians, viewpoint_cams=viewpoint_batch, return_stats=True, debug_loss=False)
            loss_sophia_tr = torch.tensor(loss_sophia_tr)
            # print(f"Iteration {iteration}, Sophia TR loss: {loss_sophia_tr.item():.10f}")

            image_sophia_tr = batch_stats[0]["images"][0]
            visibility_filter = batch_stats[0]["visibility_filter"]
            radii = batch_stats[0]["max_radii"]
            viewspace_point_tensor = batch_stats[0]["viewspace_point_tensor"]

            sophia_update_start.record()
            s_sophia_tr = sophia_optimizer.get_update(g, JTJv_func1, Dhat_func1, z_gen_func, S)
            s_sophia_tr_old = s_sophia_tr.clone()

            clip_start.record()
            # s_sophia_tr = clip_kl(gaussians, s_sophia_tr, opt.kl_threshold, 0.2, lr)
            s_sophia_tr = clip_hellinger(gaussians, s_sophia_tr, opt.kl_threshold, lr)

            # DEBUG
            if iteration % gif_interval == 0:
                debug_losses = []
                for i, vc in enumerate(viewpoint_stack):
                # for i, vc in enumerate(test_stack):
                    if i >= len(sophia_images):
                        break
                    debug_loss_i, debug_batch_stats_i = loss_func(gaussians=gaussians, viewpoint_cams=[vc], return_stats=True)
                    debug_losses.append(debug_loss_i.item())

                    sophia_image_i = debug_batch_stats_i[0]["images"][0]
                    sophia_images[i].append(sophia_image_i.detach())
                    sophia_losses[i].append(debug_loss_i.item())
                print(f"Debug losses at iteration {iteration}: {debug_losses}")
                print(f"Debug loss sum: {sum(debug_losses)}")
            # DEBUG END

            adam_start.record()
            s_adam = adam_optimizer.get_update(g)

            if opt.use_adam:
                # print("Using Adam optimizer step")
                s_sophia_tr = s_adam

            # if iteration % 200 == 1:
            if False and iteration == 6000:
                safe_interact(local=locals(), banner="Debug prompt at iteration 6000")

            loss = loss_sophia_tr
            Ll1 = torch.tensor(0.0)
            Ll1depth = torch.tensor(0.0)

            iter_end.record()
            torch.cuda.synchronize()

            # print("sophia step")
            # debug_hellinger(gaussians, s_sophia_tr, debug=False)
            # print("adam step")
            # debug_hellinger(gaussians, s_adam)
            # print("loss at iter {}: {}".format(iteration, loss.item()))
            # safe_interact(local=locals(), banner="Debug prompt before applying update")

            # Progress bar
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp), train_test_exp, val_indices=None, test_stack=test_stack, train_stack=viewpoint_stack)

            post_step_start.record()

            # First update
            if iteration < opt.iterations:
                gaussians.update_step(s_sophia_tr)

                if not opt.use_adam and opt.normalize_rotation and iteration % opt.normalize_rotation_interval == 0:
                    gaussians._rotation /= gaussians._rotation.norm(dim=1, keepdim=True)

            # Densification
            if iteration < opt.densify_until_iter:

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    # TODO: Fix densification criterion to be using gradient of opacity
                    # TODO: Reset momentum of optimizers after densification
                    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                    # dead_mask = (g.opacity <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=dataset.cap_max, growth_factor=1.05)

                    sophia_optimizer.reset_indices(dead_mask)
                    adam_optimizer.reset_indices(dead_mask)
                    s_sophia_tr.reset_indices_(dead_mask)

                    prune_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")

                    sophia_optimizer.densify_and_prune(prune_mask)
                    adam_optimizer.densify_and_prune(prune_mask)
                    s_sophia_tr.densify_and_prune_(prune_mask)

                    print(f"Num dead gaussians at iteration {iteration}: {dead_mask.sum().item()}")
                    print(f"After densification at iteration {iteration}, total gaussians: {gaussians.get_xyz.shape[0]}")


                    # safe_interact(local=locals(), banner=f"After densification at iteration {iteration} prompt")

            # Inject noise last
            if iteration < opt.iterations:
                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)

                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                print("Model path: {}".format(dataset.model_path + "/chkpnt" + str(iteration) + ".pth"))
                torch.save((gaussians.capture(), iteration), dataset.model_path + "/chkpnt" + str(iteration) + ".pth")
                # import code; code.interact(local=locals(), banner="Debug prompt after saving")

            post_step_end.record()

            torch.cuda.synchronize()

            iter_time = iter_start.elapsed_time(iter_end)
            setup_time = iter_start.elapsed_time(gradient_start)
            gradient_time = gradient_start.elapsed_time(sophia_update_start)
            sophia_update_time = sophia_update_start.elapsed_time(clip_start)
            clip_time = clip_start.elapsed_time(adam_start)
            adam_time = adam_start.elapsed_time(iter_end)
            post_step_time = post_step_start.elapsed_time(post_step_end)

            if tb_writer:
                tb_writer.add_scalar('timings/iteration_time', iter_time, iteration)
                tb_writer.add_scalar('timings/setup_time', setup_time, iteration)
                tb_writer.add_scalar('timings/gradient_time', gradient_time, iteration)
                tb_writer.add_scalar('timings/sophia_update_time', sophia_update_time, iteration)
                tb_writer.add_scalar('timings/clip_time', clip_time, iteration)
                tb_writer.add_scalar('timings/adam_time', adam_time, iteration)
                tb_writer.add_scalar('timings/post_step_time', post_step_time, iteration)

    # print("\n[ITER {}] Saving Gaussians".format(iteration))
    # scene.save(iteration)

    # gif_renderer = GifRenderer(num_rows=1, num_cols=2, figsize=(10, 6), gif_interval=1)
    # gif_renderer.add_gt(0, 0, viewpoint_cam.original_image)
    # gif_renderer.add_series(0, 1, sophia_images, sophia_losses, title="Sophia TR (Ours)")
    # gif_renderer.animate(f"figures/train_sophia_tr.gif", interval=500)
    # print(f"save figures/train_sophia_tr.gif")

    gif_renderer = GifRenderer(num_rows=len(sophia_images), num_cols=2, figsize=(10, 14), gif_interval=1)
    for row in range(len(sophia_images)):
        gif_renderer.add_gt(row, 0, viewpoint_stack[row].original_image)
        gif_renderer.add_series(row, 1, sophia_images[row], sophia_losses[row], title="Sophia TR (Ours)")
    fname = f"figures/train_sophia_tr.gif" if not opt.use_adam else f"figures/train_adam.gif"
    gif_renderer.animate(fname, interval=500)
    print(f"save {fname}")


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, val_indices=None, test_stack=None, train_stack=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if val_indices is None:
            num_val_images = 20
            val_stride = max(1, len(train_stack) // num_val_images)
            val_indices = list(range(0, len(train_stack), val_stride))
        validation_configs = ({'name': 'test', 'cameras' : test_stack}, 
                              {'name': 'train', 'cameras' : [train_stack[idx] for idx in val_indices]} )
        print(f"\n[ITER {iteration}] val_indices: {val_indices}")

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def prepare_output_and_logger(args, opt):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        cfg_log_f.write('\n')
        cfg_log_f.write(str(Namespace(**vars(opt))))
        cfg_log_f.write('\n')

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1] + list(range(0, 30001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gif_interval", type=int, default=1000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.gif_interval)

    # All done
    print("\nTraining complete.")
