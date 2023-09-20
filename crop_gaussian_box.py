import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement

class GaussianModel2:

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = np.empty(0)
        self._features_dc = np.empty(0)
        self._features_rest = np.empty(0)
        self._scaling = np.empty(0) 
        self._rotation = np.empty(0)
        self._opacity = np.empty(0)
        self.max_radii2D = np.empty(0)
        self.xyz_gradient_accum = np.empty(0)
        self.denom = np.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

    def load_gaussian_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = xyz
        self._features_dc = features_dc.transpose(0, 2, 1)
        self._features_rest = features_extra.transpose(0, 2, 1)
        self._opacity = opacities
        self._scaling = scales
        self._rotation = rots
        self.active_sh_degree = self.max_sh_degree

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_gaussian_ply(self, path):
        xyz = self._xyz
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.transpose(0, 2, 1).reshape((self._features_dc.shape[0], self._features_dc.shape[1] * self._features_dc.shape[2]))
        f_rest = self._features_rest.transpose(0, 2, 1).reshape((self._features_rest.shape[0], self._features_rest.shape[1] * self._features_rest.shape[2]))
        opacities = self._opacity
        scale = self._scaling
        rotation = self._rotation

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def crop_pointclouds(self):
        point_num = self._xyz.shape[0]
        xyz_new = []
        f_dc_new = []
        f_rest_new = []
        opacities_new = []
        scale_new = []
        rotation_new = []
        print(f'Original number of points: {point_num}')
        for idx in range(point_num):
            if self._xyz[idx, 2] > 3 and self._xyz[idx, 2] < 15 and self._xyz[idx, 1] > -9:
                xyz_new.append(self._xyz[idx:idx+1, :])
                f_dc_new.append(self._features_dc[idx:idx+1, :, :])
                f_rest_new.append(self._features_rest[idx:idx+1, :, :])
                opacities_new.append(self._opacity[idx:idx+1, :])
                scale_new.append(self._scaling[idx:idx+1, :])
                rotation_new.append(self._rotation[idx:idx+1, :])

        print(f'Number of points after cleaning: {len(xyz_new)}')
        self._xyz = np.concatenate(xyz_new, axis=0)
        self._features_dc = np.concatenate(f_dc_new, axis=0)
        self._features_rest = np.concatenate(f_rest_new, axis=0)
        self._opacity = np.concatenate(opacities_new, axis=0)
        self._scaling = np.concatenate(scale_new, axis=0)
        self._rotation = np.concatenate(rotation_new, axis=0)

if __name__ == "__main__":
    # Set up command line argument parser
    model = GaussianModel2(sh_degree=3)
    model.load_gaussian_ply('D:/Code/gaussian-splatting/output/colmap_2_5/point_cloud/iteration_10000/point_cloud2.ply')
    model.crop_pointclouds()
    model.save_gaussian_ply('D:/Code/gaussian-splatting/output/colmap_2_5/point_cloud/iteration_10000/point_cloud.ply')
    
