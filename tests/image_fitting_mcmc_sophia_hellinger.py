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
import json
import math
import numpy as np
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state, safe_interact, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
from PIL import Image
from functools import partial
from scene.gaussian_model import build_scaling_rotation
from solver.gaussian_model_vector import GaussianModelVector
from solver.adam_optimizer import AdamOptimizer
from solver.sophia_optimizer import SophiaOptimizer
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy, construct_Dhat_func
from solver.hellinger_clip import clip_hellinger, debug_hellinger

from image_fitting_utils import get_image_name

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def init_uniform_gaussians(num_points, sh_degree, opt):
    gaussians = GaussianModel(sh_degree, opt.optimizer_type)

    bd = 2
    xyz = bd * (torch.rand(num_points, 3, device="cuda") - 0.5)
    scales = torch.rand(num_points, 3, device="cuda")
    d = 3
    features_dc = torch.rand(num_points, 1, d, device="cuda")
    features_rest = torch.zeros(num_points, 15, 3, device="cuda")
    u = torch.rand(num_points, 1, device="cuda")
    v = torch.rand(num_points, 1, device="cuda")
    w = torch.rand(num_points, 1, device="cuda")
    quats = torch.cat(
        [torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
         torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
         torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
         torch.sqrt(u) * torch.cos(2.0 * math.pi * w), ], -1,)
    opacities = torch.ones((num_points, 1), device="cuda")

    gaussians._xyz = nn.Parameter(xyz.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
    gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(True))
    gaussians._scaling = nn.Parameter(scales.requires_grad_(True))
    gaussians._rotation = nn.Parameter(quats.requires_grad_(True))
    gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    gaussians.exposure_mapping = {"image": 0}
    gaussians.pretrained_exposures = None
    exposure = torch.eye(3, 4, device="cuda")[None].repeat(1, 1, 1)
    gaussians._exposure = nn.Parameter(exposure.requires_grad_(True))
    gaussians.training_setup(opt)

    return gaussians

def build_camera(image_path):
    image_pil = Image.open(image_path)
    W, H = image_pil.size
    viewmat = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 8,
                        0, 0, 0, 1], dtype=np.float32).reshape(4, 4)
    fx = math.pi / 2.0
    FoVx = 0.5 * float(W) / math.tan(0.5 * fx)
    FoVy = FoVx * H / W
    camera = Camera(resolution=(W, H), 
                    colmap_id=0, 
                    R=viewmat[:3, :3], T=viewmat[:3, 3], 
                    FoVx=FoVx, FoVy=FoVy, 
                    depth_params=None, 
                    image=image_pil, 
                    invdepthmap=None, 
                    image_name="image", 
                    uid=0, 
                    data_device="cuda",)
    return camera


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, image_path, num_points=1000):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()

    testing_iterations = testing_iterations + list(range(0, opt.iterations + 1, opt.eval_interval))

    cameras = [build_camera(image_path)]

    ####### Some fixed parameters #########
    num_images = 1
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    image_name = get_image_name(image_path)
    model_path = ""
    ####### Some fixed parameters #########

    if model_path != "":
        os.makedirs(model_path, exist_ok=True)

    tb_writer = prepare_output_and_logger(args)

    first_iter = 0
    sh_degree = 3
    gaussians = init_uniform_gaussians(num_points, sh_degree, opt)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    kl_threshold_func = get_expon_lr_func(lr_init=opt.kl_threshold_init, 
                                          lr_final=opt.kl_threshold_final, 
                                          lr_delay_mult=opt.kl_threshold_delay_mult,
                                          max_steps=opt.iterations)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = cameras
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    lr = GaussianModelVector(xyz=opt.xyz_lr_init,  
                             features_dc=opt.features_dc_lr, 
                             features_rest=opt.features_rest_lr,
                             scaling=opt.scaling_lr,
                             rotation=opt.rotation_lr,
                             opacity=opt.opacity_lr,
                             exposure=opt.exposure_lr,
                             gaussians=gaussians)
    adam_optimizer = AdamOptimizer(lr=lr, betas=(opt.adam_beta1, opt.adam_beta2), eps=1e-15, clip=False)
    adam_optimizer.reset()

    sophia_optimizer = SophiaOptimizer(lr=lr, 
                                       betas=(opt.adahessian_beta1, opt.adahessian_beta2),
                                       eps=1e-20, clip=False,
                                       gamma=opt.sophia_gamma,
                                       diagonal_update_interval=opt.diagonal_update_interval,
                                       num_init_iter=opt.diagonal_init_iter,
                                       num_init_restart_iter=opt.diagonal_init_restart_iter,
                                       num_update_iter=opt.diagonal_update_iter,
                                       num_update_restart_iter=opt.diagonal_update_restart_iter
                                       )
    sophia_optimizer.reset()

    if opt.use_adam:
        safe_interact(local=locals(), banner="Using Adam optimizer - not Sophia")

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        lr = GaussianModelVector(xyz=xyz_lr,  
                                 features_dc=opt.features_dc_lr, 
                                 features_rest=opt.features_rest_lr,
                                 scaling=opt.scaling_lr,
                                 rotation=opt.rotation_lr,
                                 opacity=opt.opacity_lr,
                                 exposure=opt.exposure_lr,
                                 gaussians=gaussians)
        adam_optimizer.update_lr(lr)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        loss.backward()
        g = GaussianModelVector.from_gaussians_grad(gaussians)

        iter_end.record()

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
                       "scale_reg": opt.scale_reg,
                       "opacity_reg": opt.opacity_reg,
                       }

        loss_func = construct_loss_func(**render_args)
        g_func = construct_g_func(**render_args)
        JTJv_func = construct_JTJv_func(**render_args)
        Dhat_func = construct_Dhat_func(**render_args)
        z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians)


        JTJv_func1 = partial(JTJv_func, gaussians=gaussians, viewpoint_cams=[viewpoint_cam], S=None, scale=1)
        Dhat_func1 = partial(Dhat_func, gaussians=gaussians, viewpoint_cams=[viewpoint_cam])

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, gaussians, cameras, render, (pipe, background))

            # Optimizer step
            if iteration < opt.iterations:
                s_adam = adam_optimizer.get_update(g)
                if opt.use_adam and opt.disable_sophia_if_use_adam:
                    s_sophia = GaussianModelVector.zeros_like(gaussians)
                else:
                    s_sophia = sophia_optimizer.get_update(g, JTJv_func1, Dhat_func1, z_gen_func, S=None)
                    s_sophia_old = s_sophia.clone()
                    kl_threshold = kl_threshold_func(iteration)
                    s_sophia = clip_hellinger(gaussians, s_sophia, kl_threshold, lr, quat_norm_tr=opt.quat_norm_tr)

                if opt.use_adam:
                    s = s_adam
                else:
                    s = s_sophia
                    s.opacity = s_adam.opacity

                if True and iteration == 6000:
                    safe_interact(local=locals(), banner="Debug optimizer step")

                gaussians.update_step(s)
                gaussians.optimizer.zero_grad(set_to_none = True)

            densified = False
            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                densified = True

                if opt.sparsify_gaussians:
                    opacity_cutoff = 0.005
                    dead_mask = (gaussians.get_opacity <= opacity_cutoff).squeeze(-1)
                    probs = 1 - (gaussians.get_opacity[:, 0]) 
                    probs[dead_mask] = 0.0
                    num_alives = (probs > 0).sum().item()
                    densify_num_samples = int(opt.sparsify_ratio * num_alives)
                    print(f"Densification sampling {densify_num_samples} gaussians among {num_alives} alives.")
                    if densify_num_samples >= 1:
                        sampled_indices, ratio = gaussians._sample_alives(probs, densify_num_samples)
                        gaussians._opacity[sampled_indices] = -10000.0

                opacity_cutoff = 0.005
                dead_mask = (gaussians.get_opacity <= opacity_cutoff).squeeze(-1)

                if opt.densify_preserve_gaussians:
                    gaussians.relocate_gs2(dead_mask=dead_mask)
                    gaussians.add_new_gs2(cap_max=args.cap_max)
                else:
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=args.cap_max)

                adam_optimizer.reset_indices(dead_mask)
                sophia_optimizer.reset_indices(dead_mask)

                prune_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                adam_optimizer.densify_and_prune(prune_mask)
                sophia_optimizer.densify_and_prune(prune_mask)

                # print(f"Num dead gaussians at iteration {iteration}: {dead_mask.sum().item()}")
                # print(f"After densification at iteration {iteration}, total gaussians: {gaussians.get_xyz.shape[0]}")

            # Optimizer step
            if iteration < opt.iterations:
                # gaussians.optimizer.step()
                # gaussians.optimizer.zero_grad(set_to_none = True)

                # s_adam_ref = GaussianModelVector.from_gaussians(gaussians) - GaussianModelVector.from_gaussians(gaussians_copy)

                # if densified:
                #     safe_interact(local=locals(), banner="Debug optimizer step")

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
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

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, gaussians, cameras, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : cameras}, 
                              {'name': 'train', 'cameras' : [cameras[idx % len(cameras)] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
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
            tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--image_path", type=str, required=True, help="Path to the training image.")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of Gaussians to use.")
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.image_path, args.num_points)

    # All done
    print("\nTraining complete.")
