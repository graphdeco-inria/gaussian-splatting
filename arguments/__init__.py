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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.cap_max = -1
        self.init_type = "random"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_preserve_gaussians = False
        self.sparsify_gaussians = False
        self.sparsify_ratio = 0.01
        self.densify_start_opacity = 0.01
        self.densify_position_noise = 0.1
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"

        self.reset_optimizer = False

        self.jvp_start = 15_001
        self.num_images = 5

        self.loss_type = "l1"
        self.huber_delta = 1e-1
        self.disable_ssim = False

        self.naive_densification = False

        self.linesearch_alpha = 1e-0
        self.linesearch_alpha_min = 1e-2
        self.linesearch_gs_min = 1e-12
        self.linesearch_alpha_decrease = 0.8
        self.linesearch_alpha_increase = 1.2
        self.linesearch_alpha_c = 0.01
        self.linesearch_force_minstep = True
        self.linesearch_val_images = 1.0

        self.damp_alpha_max = 0.2
        self.damp_alpha_min = 1e-2
        self.damp_increase = 1.5
        self.damp_increase_high = 10.0
        self.damp_decrease = 0.6

        self.pixel_sample_rate_max = 1.0
        self.pixel_sample_rate_min = 1.0
        self.pixel_sample_rate = self.pixel_sample_rate_max
        self.pixel_sample_rate_increase = 1.2
        self.pixel_sample_rate_decrease = 0.9

        self.splat_sample_update_freq = 20
        self.splat_sample_rate = 1.0

        self.pcg_num_iter = 1
        self.pcg_restart_iter = 5
        self.pcg_tol = 1e-15

        self.preconditioner_use_adam_variance = False
        self.use_preconditioner = True
        self.preconditioner_image_batch_size = 5
        self.preconditioner_reset = True
        self.preconditioner_reset_iter = 20
        self.preconditioner_warmup_from_gradient_samples = False
        self.preconditioner_warmup_interval = 10
        self.preconditioner_warmup_iter = 2
        self.sophia_gamma = 1.0
        self.sophia_epsilon = 1.0

        self.use_adam = False
        self.use_adam_yes = False                   # Disable interactive session before using adam
        self.disable_sophia_if_use_adam = True
        self.enable_adam_tr = False

        self.adahessian_beta1 = 0.9
        self.adahessian_beta2 = 0.999
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999

        self.eval_interval = 10

        scale_const = 1e0

        # self.xyz_scale = 1e-3 * scale_const * 1.0
        # self.features_dc_scale = 1e-3 * scale_const * 1.0
        # self.features_rest_scale = 1e-3
        # self.scaling_scale = 1e-3 * scale_const * 1e0 * 1.0
        # self.rotation_scale = 1e-3 * scale_const * 1e0 * 1.0
        # self.opacity_scale = 1e-3 * scale_const * 1.0
        # self.exposure_scale = 1e-3 * scale_const * 1.0

        # scale is related to preconditioning the solver
        self.xyz_scale = 1.6e-4 * scale_const * 1.0
        self.features_dc_scale = 2.5e-3 * scale_const * 1.0
        self.features_rest_scale = self.features_dc_scale / 20.0
        self.scaling_scale = 5e-3 * scale_const * 1e1 * 1.0
        self.rotation_scale = 1e-3 * scale_const * 1e0 * 1.0
        self.opacity_scale = 2.5e-2 * scale_const * 1.0
        self.exposure_scale = 1.0 * scale_const * 1.0

        # # lr is the truncated learning rate used in optimization
        # self.xyz_lr_init = 1.0
        # self.xyz_lr_final = 1.0
        # self.xyz_lr_max_steps = self.iterations
        # self.features_dc_lr = 1.0
        # self.features_rest_lr = 1.0
        # self.scaling_lr = 1.0
        # self.rotation_lr = 1.0
        # self.opacity_lr = 1.0
        # self.exposure_lr = 1.0

        # self.lr_scale = 10.0
        self.lr_scale = 1.0
        self.xyz_lr_init = 1.6e-4 * scale_const * self.lr_scale
        self.xyz_lr_final = 1.6e-6 * scale_const * self.lr_scale
        self.xyz_lr_decay = 0.2 # 0.996
        self.xyz_lr_max_steps = self.iterations

        # lr is the truncated learning rate used in optimization
        self.features_dc_lr = 2.5e-3 * scale_const * 1.0 * self.lr_scale
        self.features_rest_lr = self.features_dc_scale / 20.0
        self.scaling_lr = 5e-3 * scale_const * 1e-0 * self.lr_scale
        self.rotation_lr = 1e-3 * scale_const * 1e-0 * self.lr_scale
        self.opacity_lr = 2.5e-2 * scale_const * 1.0 * self.lr_scale
        self.exposure_lr = 1.0 * scale_const * 1.0

        # NOTE: damp needs to be relative to scale^2
        self.damp_init = 1e-9 * (scale_const ** 2)      
        self.damp_min = 1e-9 * (scale_const ** 2)       
        self.damp_max = 1e-2 * (scale_const ** 2)       
        self.damp_res_target = 1e-4
        self.damp = self.damp_init

        self.quat_norm_tr = 0.01

        self.noise_opacity_thresh = 0.995
        self.noise_lr = 5e5
        self.scale_reg = 0.01
        self.opacity_reg = 0.01
        self.binarize_opacity_reg = False
        self.color_reg = 0.0

        self.regularize_scaling = False
        self.scaling_reg_weight = 5e-3
        self.scaling_reg_thresh = 5
        self.debug_loss = False

        self.kl_threshold_init = 1e-6
        self.kl_threshold_final = 1e-8
        self.kl_threshold_delay_mult = 0.01
        self.kl_threshold_max_steps = 30_000
        self.opacity_threshold_scale = 1.0
        self.diagonal_init_iter = 20
        self.diagonal_init_restart_iter = 3
        self.diagonal_update_iter = 1
        self.diagonal_update_restart_iter = 1
        self.diagonal_update_interval = 5
        self.diagonal_accum_abs = False

        self.normalize_rotation = True
        self.normalize_rotation_interval = 10

        self.opacity_prune_thresh = 0.005

        self.eval_interval = 1000

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
