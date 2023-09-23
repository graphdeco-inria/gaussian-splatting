import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

from PIL import Image
from tqdm import tqdm

import open3d as o3d
import matplotlib.pyplot as plt

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

class camera:
    def __init__(self,K,R,T,name='cam',width=872,height=490) -> None:
        self.K = K
        self.R = R
        self.T = T
        self.width= width
        self.height = height
        self.camera_position = -R.T@T
        self.name = name
        self.frustum_vertices = []

    def get_view_frustum(self,near_clip=0.01,far_clip=20):
        K = self.K
        # 相机内部参数
        fx, fy, cx, cy = K

        # 图像分辨率
        R = self.R
        T = self.T
        width = self.width
        height = self.height
        # 相机的外部参数（相机的位置和方向）
        # 这里只是一个示例，你需要提供实际的相机外部参数
        camera_position = self.camera_position  # 相机位置
        

        # 创建一个点云表示相机位置
        camera_point = o3d.geometry.PointCloud()
        camera_point.points = o3d.utility.Vector3dVector([camera_position])
        camera_point.paint_uniform_color([1, 0, 0])  # 设置点的颜色为红色

        # 创建表示相机视锥体的线集
        line_set = o3d.geometry.LineSet()

        # 计算相机视锥体的八个顶点

        # 视锥体的四个角点在相机坐标系下的坐标
        top_left = np.array([(0 - cx) * near_clip / fx, (0 - cy) * near_clip / fy, near_clip])
        top_right = np.array([(width - cx) * near_clip / fx, (0 - cy) * near_clip / fy, near_clip])
        bottom_left = np.array([(0 - cx) * near_clip / fx, (height - cy) * near_clip / fy, near_clip])
        bottom_right = np.array([(width - cx) * near_clip / fx, (height - cy) * near_clip / fy, near_clip])

        # 通过相机的旋转将角点变换到世界坐标系
        vertices = [np.dot(R.T, top_left) + camera_position,
                    np.dot(R.T, top_right) + camera_position,
                    np.dot(R.T, bottom_right) + camera_position,
                    np.dot(R.T, bottom_left) + camera_position,
                    np.dot(R.T, top_left) * (far_clip / near_clip) + camera_position,
                    np.dot(R.T, top_right) * (far_clip / near_clip) + camera_position,
                    np.dot(R.T, bottom_right) * (far_clip / near_clip) + camera_position,
                    np.dot(R.T, bottom_left) * (far_clip / near_clip) + camera_position]

        # 定义相机视锥体的边
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]]

        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1] for _ in range(len(lines))]))  # 设置线的颜色为蓝色

        colors = [[0, 0, 1] for i in range(len(lines))]
        vertices_array = np.array(vertices)
        line_mesh1 = LineMesh(vertices_array, lines, colors, radius=0.05)
        line_mesh1_geoms = line_mesh1.cylinder_segments
        self.frustum_vertices = vertices
        
        near_plane = [self.frustum_vertices[0], self.frustum_vertices[1], self.frustum_vertices[2], self.frustum_vertices[3]]
        far_plane = [self.frustum_vertices[4], self.frustum_vertices[5], self.frustum_vertices[6], self.frustum_vertices[7]]
        left_plane = [self.frustum_vertices[0], self.frustum_vertices[3], self.frustum_vertices[7], self.frustum_vertices[4]]
        right_plane = [self.frustum_vertices[1], self.frustum_vertices[5], self.frustum_vertices[6], self.frustum_vertices[2]]
        top_plane = [self.frustum_vertices[0], self.frustum_vertices[1], self.frustum_vertices[5], self.frustum_vertices[4]]
        bottom_plane = [self.frustum_vertices[3], self.frustum_vertices[2], self.frustum_vertices[6], self.frustum_vertices[7]]

        self.planes = [near_plane, far_plane, left_plane, right_plane, top_plane, bottom_plane]
        self.planes_bool = [True,False,True,True,False,True]
        self.planes_coeffs = []
        for plane in self.planes:
        # 计算平面方程 Ax + By + Cz + D = 0 中的 ABCD 系数
            self.planes_coeffs.append(self.plane_equation(plane))

        return camera_point, line_mesh1_geoms

    def frustum_check(self,point):
        if not self.frustum_vertices:
            self.get_view_frustum() 
        for coeff,is_positive in zip(self.planes_coeffs,self.planes_bool):   
            A,B,C,D = coeff
            distance = A * point[0] + B * point[1] + C * point[2] + D
            #print(distance)
            if (is_positive and distance < 0) or (not is_positive and distance > 0):
                return False
        return True
    
    def visible_check(self,p,voxel_grid):
        in_frustum = self.frustum_check(p)
        if not in_frustum:
            return False

        aabb = voxel_grid.get_axis_aligned_bounding_box()
        aabb = aabb.get_box_points()
        aabb = np.asarray(aabb)
        x_min = np.min(aabb[:,0])
        x_max = np.max(aabb[:,0])

        y_min = np.min(aabb[:,1])
        y_max = np.max(aabb[:,1])

        z_min = np.min(aabb[:,2])
        z_max = np.max(aabb[:,2])

        min_v = np.array([x_min,y_min,z_min])
        max_v = np.array([x_max,y_max,z_max])
        output = False
        sample_density = 100
        
        if (p >= min_v).all() and (p <= max_v).all():
            line_queries = np.linspace(p, self.camera_position, sample_density)
            output_p = voxel_grid.check_if_included(o3d.utility.Vector3dVector(line_queries))
            if any(output_p):
                output = False
            else:
                output = True
        else:
            output = True
        return output
    
    @staticmethod
    def plane_equation(plane):
        points = np.array(plane)
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1,v2)
        A,B,C = normal
        D = -np.dot(normal, points[0])

        return A, B, C, D
    
def get_aabb(cameras):
    vertices = []
    for cam in cameras:
        vertices += cam.frustum_vertices
    vertices = np.asarray(vertices)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (0, 1, 0)
    points = aabb.get_box_points()
    points = np.asarray(points)
    return aabb,points

def get_obb(cameras):
    vertices = []
    for cam in cameras:
        vertices += cam.frustum_vertices
    vertices = np.asarray(vertices)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    aabb = pcd.get_oriented_bounding_box()
    aabb.color = (0, 1, 0)
    points = aabb.get_box_points()
    points = np.asarray(points)
    return aabb,points

def create_aabb_voxel(aabb_points,voxel_size=5):
    # 计算AABB的最小和最大顶点坐标
    min_x = min([point[0] for point in aabb_points])
    min_y = min([point[1] for point in aabb_points])
    min_z = min([point[2] for point in aabb_points])

    max_x = max([point[0] for point in aabb_points])
    max_y = max([point[1] for point in aabb_points])
    max_z = max([point[2] for point in aabb_points])

    # 计算AABB框的尺寸
    aabb_width = max_x - min_x
    aabb_height = max_y - min_y
    aabb_depth = max_z - min_z

    # 计算每个轴上的体素数量
    num_x_voxels = int(aabb_width / voxel_size)
    num_y_voxels = int(aabb_height / voxel_size)
    num_z_voxels = int(aabb_depth / voxel_size)

    # 生成体素中心点坐标
    voxel_centers = []
    for i in range(num_x_voxels):
        for j in range(num_y_voxels):
            for k in range(num_z_voxels):
                x = min_x + i * voxel_size + voxel_size / 2
                y = min_y + j * voxel_size + voxel_size / 2
                z = min_z + k * voxel_size + voxel_size / 2
                voxel_centers.append([x, y, z])
    voxel_centers = np.asarray(voxel_centers)
    # 将体素中心点坐标转换为Open3D的体素格式
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)
    #point_cloud.paint_uniform_color([0.5,0.5,0.5])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
    return voxel_centers, voxel_grid

def get_camera(txt_dir):
    img_txt = os.path.join(txt_dir,'images.txt')
    cam_txt = os.path.join(txt_dir, 'cameras.txt')
    with open(img_txt,'r') as f:
        ll = f.readlines()
    ll = ll[4:]

    ll = ll[::2]
    # target_name = target_name.split('.')[-2]
    cameras = []
    for l in ll:
        cont = l.split()
        cam_name = os.path.basename(cont[9])
        cam_name = cam_name[:-4]
        rt = cont
        cam_ind = int(cont[8])
        cam_name = os.path.basename(rt[9]).split('.')[0]

        #QW, QX, QY, QZ = [float(rt[1]), float(rt[2]),  float(rt[3]),  float(rt[4])]
        Rq1 = np.asarray([float(rt[2]),  float(rt[3]),  float(rt[4]), float(rt[1])])
        r1 = R.from_quat(Rq1)
        Rm1 = r1.as_matrix()
        T = np.asarray([float(rt[5]), float(rt[6]), float(rt[7])])

        with open(cam_txt,'r') as f:
            ll = f.readlines()
            ll = ll[3:]
        K = ll[cam_ind - 1]
        K = K.split()[4:]
        K = [float(i) for i in K]
        cameras.append(camera(K,Rm1,T,cam_name))
    return cameras

def check_voxel(voxel_grid, queries, voxel_size=1):
    #point cloud is the center of the voxel
    output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    colors = np.array([[1, 0, 0] if i else [0.83,0.83,0.83] for i in output])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(queries)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud, output

def check_visible_overlap(cameras, queries, voxel_grid, occu_list = [],filter_thres = 10):
    
    vertices = []
    for cam in cameras:
        vertices += cam.frustum_vertices
    max_view = len(cameras)
    filter_thres = min(filter_thres,max_view)
    vertices = np.array(vertices)
    
    x_min = np.min(vertices[:,0])
    x_max = np.max(vertices[:,0])

    y_min = np.min(vertices[:,1])
    y_max = np.max(vertices[:,1])

    z_min = np.min(vertices[:,2])
    z_max = np.max(vertices[:,2])

    min_v = np.array([x_min,y_min,z_min])
    max_v = np.array([x_max,y_max,z_max])
    output_list = []
    
    for ind,p in enumerate(queries):
        if occu_list and occu_list[ind]:
            output_list.append(filter_thres)
            continue
        if (p >= min_v).all() and (p <= max_v).all():
            count = 0
            for cam in cameras:
                if cam.visible_check(p,voxel_grid):
                    count += 1
            #print(count)
            output_list.append(count)
        else:
            output_list.append(0)
    cmap = plt.cm.get_cmap('plasma')
    colors = np.array([cmap(i/max_view)[:3] if i>0 else [0.83,0.83,0.83] for i in output_list])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(queries)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    #fitler
    filter_points = np.array([p for ind,p in enumerate(queries) if output_list[ind]>filter_thres]) 
    filter_pcd = o3d.geometry.PointCloud()
    filter_pcd.points = o3d.utility.Vector3dVector(filter_points)
    filter_pcd.paint_uniform_color([0.8, 0, 0])

    return point_cloud, filter_pcd, output_list,

if __name__ == '__main__':

    txt_dir = './res-2-0-0/sparse/0'
    pcd_dir = './res-2-0-0/sparse/0/points3D.ply'
    cameras = get_camera(txt_dir)

    cam_points = []
    frustums = []
    for cam in cameras:
        p,frustum_line = cam.get_view_frustum()
        cam_points.append(p)
        frustums+= frustum_line

    aabb,aabb_p = get_aabb(cameras)
    center,aabb_voxel_grid = create_aabb_voxel(aabb_p,voxel_size=1)

    pcd = o3d.io.read_point_cloud(pcd_dir)
    pcd = pcd.crop(aabb)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.8)

    occu_pcd, occu_list = check_voxel(voxel_grid,center)
    overlap_pcd, filter_pcd, overlap_list = check_visible_overlap(cameras, center, voxel_grid, occu_list)
    
    overlap_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(overlap_pcd,voxel_size=0.15)
    filter_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(filter_pcd,voxel_size=0.5)
    
    all_vis = [voxel_grid, aabb,overlap_voxel_grid, filter_voxel_grid] + cam_points + frustums 
    o3d.visualization.draw_geometries(all_vis)

    file_path = "./occu_voxel.ply"  # 保存路径
    o3d.io.write_voxel_grid(file_path, voxel_grid)

    file_path = "./filter_voxel.ply"  # 保存路径
    o3d.io.write_voxel_grid(file_path, filter_voxel_grid)


