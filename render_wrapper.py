from argparse import ArgumentParser, Namespace
import os

import numpy as np
import torch
import torchvision

from arguments import GroupParams, ModelParams
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from scene.cameras import Camera_Simple
from utils.general_utils import safe_state


class GS_Model():
    def __init__(self, model_path="/home/cviss/PycharmProjects/gaussian-splatting/output/1e5592be-5"):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        self.model_path = model_path
        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)

        args = self.get_combined_args(model_path)

        # Initialize system state (RNG)
        safe_state(True)

        dataset = model.extract(args)

        '''This replaces the PipelineParams Object'''
        self.pipeline = GroupParams()
        self.pipeline.compute_cov3D_python = False
        self.pipeline.convert_SHs_python = False
        self.pipeline.debug = False

        self.gaussians = GaussianModel(3)
        self.scene = Scene(dataset, self.gaussians, load_iteration=-1, shuffle=False)
        self.bg_color = [1, 1, 1]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

    def render_view(self, R_mat, T_vec, img_width, img_height, save=False):
        '''
        Call this method to render a new view from the scene by inputting new pose and desired image width and height
        Note: The FoVx and FoVy can be changed please see utils.graphics_utils.fov2focal and focal2fov methods
        @param R_mat: 3x3 camera extrinsic rotation matrix numpy array
        @param T_vec: 3, camera extrinsic translation vector numpy array
        @param img_width: image width pixels
        @param img_height: image height pixels
        @return rendered PIL image
        '''
        view = Camera_Simple(colmap_id=0, R=R_mat, T=T_vec, img_width=img_width, img_height=img_height,
                             FoVx=1.0, FoVy=1.0, uid=None)

        rendering = render(view, self.gaussians, self.pipeline, self.background)["render"]
        rendering = torchvision.transforms.ToPILImage()(rendering)
        if save:
            #torchvision.utils.save_image(rendering, "test.png")
            rendering.save('test.jpg')
        return rendering

    def get_combined_args(self, model_path):
        try:
            cfgfilepath = os.path.join(model_path, "cfg_args")
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
        except TypeError:
            print("Config file not found at")
            pass
        args_cfgfile = eval(cfgfile_string)

        merged_dict = vars(args_cfgfile).copy()
        return Namespace(**merged_dict)

#if __name__ == '__main__':
#    model_1 = GS_Model(model_path="/home/cviss/PycharmProjects/gaussian-splatting/output/1e5592be-5")
#    model_1.render_view(R_mat=np.array([[-0.8145390529478596, 0.01889517829114354, 0.5798009170915043],
#                                                      [-0.09778674725285423, 0.98069508920201, -0.16933662945969005],
#                                                      [-0.571807557911322, -0.19462814352607866, -0.7969667511653681]]),
#                        T_vec=np.array([-2.7518888678267177, 0.5298969558367272, 4.8760898433256425]),
#                        img_width=1440, img_height=1920, save=True)