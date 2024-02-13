import torch
import yaml
import os
from gaussian_renderer import render
import torchvision
from gaussian_renderer import GaussianModel
from camera_pos_utils import *

from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class DummyPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


class DummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H, C2C_Rot=np.eye(4, dtype=np.float32), C2C_T=np.eye(4, dtype=np.float32)):
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,
                                                                                                             1).cuda()
        self.R = R
        self.T = T

        world2View2 = getWorld2View2(self.R, self.T, np.array([0, 0, 0]), 1.0)

        world2View2 = C2C_Rot @ world2View2
        world2View2 = C2C_T @ world2View2

        self.world_view_transform = torch.tensor(world2View2).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy

    def get_new_pose(self):
        """
        Use this function to get the update pose cuz world_view_transform uses the OpenGL view matrix convention
        i.e.,
        [[Right_x, Right_y, Right_z, 0],
        [Up_x, Up_y, Up_z, 0],
        [Look_x, Look_y, Look_z, 0],
        [Position_x, Position_y, Position_z, 0]]
        Or [R|0]
           [T|1] so we need to rearrange before returning back to keep conventions consistent
        """
        world_view_transform = self.world_view_transform.cpu()
        pose = np.eye(4, dtype=np.float32)
        pose[0:3, 0:3] = world_view_transform[0:3, 0:3]
        pose[:3, 3] = np.transpose(world_view_transform[3, :3])
        return pose


class GS_Model():
    def __init__(self, config_path: str, device: str = "cuda:0"):
        self.images = None

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config["images_txt_path"] is not None:
            self.images = ImagesMeta(config["images_txt_path"])
            self.images_thumbnails = config["images_thumbnails"]

        self.world2custom = np.array(config["world2custom"])

        # First Set GPU Context (i.e., we can put different models on different GPUs if needed)
        device = torch.device(device)
        torch.cuda.set_device(device)

        self.pipeline = DummyPipeline()

        self.gaussians = GaussianModel(3)  # 3 is the default sh-degree
        self.gaussians.load_ply(config["ply_path"])

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    def render_view(self, cam: DummyCamera, save: bool = False, out_path: str = "./test.jpg"):
        """
        @param cam: DummyCamera object
        @param save: whether to save the image
        @param out_path: where and what to save the image as
        @return rendered PIL image
        """

        result = render(cam, self.gaussians, self.pipeline, self.background)["render"]

        result = torchvision.transforms.ToPILImage()(result)

        if save:
            result.save(out_path)

        return result


if __name__ == '__main__':
    model1 = GS_Model(
        config_path="/home/cviss/PycharmProjects/GS_Stream/output/dab812a2-1/point_cloud/iteration_30000/config.yaml")
    R_mat = np.array([[-0.70811329, -0.21124761, 0.67375813],
                      [0.16577646, 0.87778949, 0.4494483],
                      [-0.68636268, 0.42995355, -0.58655453]])
    T_vec = np.array([-0.32326042, -3.65895232, 2.27446875])

    C2C_Rot = rotate4(np.radians(90), 0, 1, 0)
    C2C_T = translate4(0, 0, 0)

    cam = DummyCamera(R=R_mat, T=T_vec, W=1600, H=1200, FoVx=1.4261863218, FoVy=1.150908963)
    print(cam.world_view_transform)

    print(model1.images.get_closest_n(cam.world_view_transform.cpu().detach().numpy()))
    # model1.render_view(cam=cam, save=True, out_path="test_roty_90_dev.jpg")
