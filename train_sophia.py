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
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy

from copy import deepcopy

from matplotlib import pyplot as plt

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

def plot_diagonal_est(v, adam_v, g):
    v = v.as_1d_tensor()
    adam_v = adam_v.as_1d_tensor()
    g = g.as_1d_tensor()
    sorted_indices = torch.argsort(adam_v)

    safe_interact(local=locals(), banner="Plotting diagonal estimates")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    splat_mask = None

    ####### Some fixed parameters #########
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    model_path = ""
    ####### Some fixed parameters #########

    pixel_sample_rate = opt.pixel_sample_rate_max
    xyz_lr = opt.xyz_lr_init
    damp = opt.damp_init

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
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
    # S = GaussianModelVector(xyz=opt.xyz_scale,
    #                         features_dc=opt.features_dc_scale,
    #                         features_rest=opt.features_rest_scale,
    #                         scaling=opt.scaling_scale,
    #                         rotation=opt.rotation_scale,
    #                         opacity=opt.opacity_scale,
    #                         exposure=opt.exposure_scale,
    #                         gaussians=gaussians)
    S = GaussianModelVector(xyz=1.0,
                            features_dc=1.0,
                            features_rest=1.0,
                            scaling=1.0,
                            rotation=1.0,
                            opacity=1.0,
                            exposure=1.0,
                            gaussians=gaussians)

    if not opt.preconditioner_use_adam_variance:
        D_est = GaussianModelVector.ones_like(gaussians)
    else:
        safe_interact(local=locals(), banner="Using Adam variance preconditioner - not AdaHessian")
    g_smoothed = 0
    g_denom_iter = 0
    D_denom_iter = 0
    D_est_smoothed = 0
    adam_D_denom_iter = 0
    adam_D_est_smoothed = 0

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    gradient_end = torch.cuda.Event(enable_timing = True)
    preconditioner_end = torch.cuda.Event(enable_timing = True)
    clip_end = torch.cuda.Event(enable_timing = True)
    update_end = torch.cuda.Event(enable_timing = True)


    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    test_viewpoint_stack = scene.getTestCameras().copy()
    viewpoint_stack = scene.getTrainCameras().copy()
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
        training_report(tb_writer, first_iter, 0.0, 0.0, l1_loss, 0.0, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, opt.jvp_start, val_indices=None)

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

        gaussians.update_learning_rate(iteration)

        num_batch_cameras = len(viewpoint_indices) if opt.num_images == -1 else min(opt.num_images, len(viewpoint_indices))
        rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
        scale = len(viewpoint_indices) / num_batch_cameras

        viewpoint_batch = []
        for rand_idx in rand_indices:
            viewpoint_cam = viewpoint_stack[rand_idx]
            viewpoint_batch.append(viewpoint_cam)

        num_val_images = int(len(viewpoint_stack) * opt.linesearch_val_images)
        val_stride = max(1, len(scene.getTrainCameras()) // num_val_images)
        val_indices = list(range(0, len(scene.getTrainCameras()), val_stride))
        val_viewpoint_stack = [viewpoint_stack[i] for i in val_indices]
        val_scale = len(viewpoint_stack) / len(val_viewpoint_stack)

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
                       "pixel_mask": None,}

        loss_func = construct_loss_func(**render_args)
        g_func = construct_g_func(**render_args)
        JTJv_func = construct_JTJv_func(**render_args)

        if opt.preconditioner_warmup_from_gradient_samples:
            random_cam_provider = CamProvider(viewpoint_batch, scale)
        else:
            random_cam_provider = CamProvider(viewpoint_stack)

        JTJv_hat_func = partial(JTJv_hat, JTJv_func=JTJv_func, gaussians=gaussians, cam_provider=random_cam_provider, batch_size=opt.preconditioner_image_batch_size, S=S, damp=damp)
        z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians)

        JTJv = partial(JTJv_func, gaussians=gaussians, viewpoint_cams=viewpoint_batch, scale=scale, S=S, damp=damp)

        start_loss, g = g_func(gaussians=gaussians, viewpoint_cams=viewpoint_batch, scale=scale, return_loss=True)

        ## ADAM ##
        g_denom_iter += 1
        beta1 = opt.adahessian_beta1
        g_smoothed = beta1 * g_smoothed + (1 - beta1) * g
        g_est = g_smoothed / (1 - beta1 ** g_denom_iter)

        gradient_end.record()

        # print("Using Adam variance for preconditioner")
        adam_D_denom_iter += 1
        beta2 = opt.adahessian_beta2
        adam_D_est_smoothed = beta2 * adam_D_est_smoothed + (1 - beta2) * (g * g)
        adam_v = (adam_D_est_smoothed / (1 - beta2 ** adam_D_denom_iter)).sqrt()

        s_adam = -(1 / (adam_v + 1e-15)) * g_est
        s_adam.clip_(-1, 1)
        s_adam = s_adam * lr

        ## ADAM ##

        if iteration == first_iter or iteration % opt.preconditioner_warmup_interval == 0:

            squared_hutchinson = False

            num_preconditioner_iters = opt.preconditioner_warmup_iter if iteration != first_iter else 200

            D_denom_iter += 1
            if squared_hutchinson:
                D_est_t = restarted_squared_hutchinson(Hz_func=JTJv_hat_func,
                                                       z_gen_func=z_gen_func,
                                                       D_init=D_est, 
                                                       # D_init=GaussianModelVector.ones_like(gaussians),
                                                       restart_iter=-1,
                                                       num_iters=num_preconditioner_iters,
                                                       damp=damp
                                                       )

                beta2 = opt.adahessian_beta2
                D_est_smoothed = beta2 * D_est_smoothed + (1 - beta2) * (D_est_t * D_est_t)
                D_est = (D_est_smoothed / (1 - beta2 ** D_denom_iter)).sqrt()
            else:
                D_precond = D_est * 1
                D_precond.clip_(damp, 1e20)

                D_est_t = restarted_hutchinson(Hz_func=JTJv_hat_func,
                                               z_gen_func=z_gen_func,
                                               # D_init=D_precond, 
                                               D_init=GaussianModelVector.ones_like(gaussians),
                                               restart_iter=-1,
                                               num_iters=num_preconditioner_iters,
                                               damp=damp
                                               )

                # safe_interact(local=locals(), banner="Hutchinson preconditioner debug")


                beta2 = opt.adahessian_beta2
                D_est_smoothed = beta2 * D_est_smoothed + (1 - beta2) * D_est_t
                D_est = (D_est_smoothed / (1 - beta2 ** D_denom_iter))

            v = opt.sophia_gamma * (D_est / (S * S))
            v.clip_(damp, 1e20)

        preconditioner_end.record()

        # plot_diagonal_est(v, adam_v, g_est)

        adam_mixed_mode = "weighted" # "weighted" "replace" "none"

        if adam_mixed_mode == "weighted":
            adam_weight = 0.1
            v = (1 - adam_weight) * v + adam_weight * adam_v
        elif adam_mixed_mode == "replace":
            v_vec = v.as_1d_tensor()
            adam_v_vec = adam_v.as_1d_tensor()
            v_vec = torch.max(v_vec, adam_v_vec)
            # v_vec = torch.min(v_vec, adam_v_vec)
            v.load_1d_tensor(v_vec)
        else:
            pass

        s_sophia = -(1 / v) * g_est
        s_sophia.clip_(-1, 1)
        s_sophia = s_sophia * lr

        if opt.use_adam:
            s_sophia = s_adam

        # safe_interact(local=locals(), banner="Before clip")

        clip_end.record()


        # print("start loss:", start_loss)

        with torch.no_grad():

            alpha = opt.linesearch_alpha

            gaussians.update_step(alpha * s_sophia)

            end_loss = loss_func(gaussians=gaussians, viewpoint_cams=viewpoint_batch, scale=scale)

            # print("end loss:", end_loss)

            iter_end.record()
            torch.cuda.synchronize()

            # loss = loss_func(gaussians=gaussians, viewpoint_cams=viewpoint_stack, scale=1.0)
            loss = 0.0
            Ll1 = 0.0

            if iteration % opt.eval_interval == 0:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, opt.jvp_start, val_indices=None)

            if tb_writer:

                tb_writer.add_scalar('train_loss_full/alpha', alpha, iteration)
                tb_writer.add_scalar('train_loss_full/pixel_sample_rate', pixel_sample_rate, iteration)
                tb_writer.add_scalar('train_loss_full/damping', damp, iteration)

            # Update lr for xyz
            lr.xyz = xyz_lr_scheduler.get_lr(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration, D_est), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
            actual_covariance = L @ L.transpose(1, 2)

            def op_sigmoid(x, k=100, x0=opt.noise_opacity_thresh):
                return 1 / (1 + torch.exp(-k * (x - x0)))
            
            xyz_noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*opt.noise_lr*lr.xyz
            xyz_noise = torch.bmm(actual_covariance, xyz_noise.unsqueeze(-1)).squeeze(-1)
            gaussians._xyz.add_(xyz_noise)

            update_end.record()
            torch.cuda.synchronize()

            gradient_time = iter_start.elapsed_time(gradient_end)
            preconditioner_time = gradient_end.elapsed_time(preconditioner_end)
            clip_time = preconditioner_end.elapsed_time(clip_end)
            update_time = iter_end.elapsed_time(update_end)
            iter_time = iter_start.elapsed_time(iter_end)

            if tb_writer:
                tb_writer.add_scalar('timings/gradient_time', gradient_time, iteration)
                tb_writer.add_scalar('timings/preconditioner_time', preconditioner_time, iteration)
                tb_writer.add_scalar('timings/clip_time', clip_time, iteration)
                tb_writer.add_scalar('timings/update_time', update_time, iteration)
                tb_writer.add_scalar('timings/iter_time', iter_time, iteration)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, jvp_start, val_indices=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    if iteration in testing_iterations or (iteration >= jvp_start):
        torch.cuda.empty_cache()
        if val_indices is None:
            num_val_images = 20
            val_stride = max(1, len(scene.getTrainCameras()) // num_val_images)
            val_indices = list(range(0, len(scene.getTrainCameras()), val_stride))
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in val_indices]} )
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
