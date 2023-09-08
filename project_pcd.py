import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

from PIL import Image
from tqdm import tqdm

import open3d as o3d


def project_depth_rgb(K1, R1, T1, rgb_A, depth_image_A):
    # open3d version
    depth_scale = 3000.0
    depth_max = 40.0
    fx1, fy1, cx1, cy1 = K1
    intrinsic_A = o3d.core.Tensor([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
    height_A, width_A = depth_image_A.shape[:2]
    depth_image_A10000 = depth_image_A * depth_scale
    depth_image_A10000 = depth_image_A10000.reshape(height_A, width_A).astype(np.uint16)
    device = o3d.core.Device('CPU:0')
    depth_image = o3d.t.geometry.Image(depth_image_A10000).to(device)
    rgb_A = rgb_A.astype(np.uint8)
    color_image = o3d.t.geometry.Image(rgb_A).to(device)
    rgbd_image = o3d.t.geometry.RGBDImage(color_image, depth_image)
    extrinsic_A = np.eye(4)
    extrinsic_A[0:3, 0:3] = R1
    extrinsic_A[0:3, 3] = T1
    extrinsic_A = o3d.core.Tensor(extrinsic_A)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                            intrinsic_A, extrinsic_A, depth_scale=depth_scale, depth_max=depth_max)
    #pcd.to_legacy()
    return pcd

def get_rt(txt_dir,target_name):
    img_txt = os.path.join(txt_dir,'images.txt')
    cam_txt = os.path.join(txt_dir, 'cameras.txt')
    with open(img_txt,'r') as f:
        ll = f.readlines()
    ll = ll[4:]

    ll = ll[::2]
    target_name = os.path.basename(target_name)
    target_name = target_name.replace('images_','')
    target_name = target_name[:-4]
    # target_name = target_name.split('.')[-2]
    for l in ll:
        cont = l.split()
        cam_name = os.path.basename(cont[9])
        cam_name = cam_name[:-4]
        if target_name == cam_name:
            #print(cam_name)
            rt = cont
            cam_ind = int(cont[8])
            break
    
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
    return Rm1, T, K


def project_pipeline(txt_dir, img, dep, masks_np):
    R1, T1, K1 = get_rt(txt_dir, img)

    depth_image_A = np.load(dep)
    rgb_A = cv2.imread(img)
    rgb_A = cv2.cvtColor(rgb_A, cv2.COLOR_BGR2RGB)
    masks_np = np.load(masks_np)
    masks_np[masks_np>0] = 1
    if len(masks_np.shape) > 2:
        mask_all = np.sum(masks_np,0)
    else:
        mask_all = masks_np
    mask_3 = np.stack((mask_all,mask_all,mask_all),2)

    rgb_A = rgb_A * mask_3
    depth_image_A = depth_image_A * mask_all
    pcd = project_depth_rgb(K1, R1, T1, rgb_A, depth_image_A)
    return pcd    

def generate_list(root):
    train_depth_dir = os.path.join(root,'train','ours_10000','depth')
    test_depth_dir = os.path.join(root,'test','ours_10000','depth')

    train_img_dir = os.path.join(root,'train','ours_10000','renders')
    test_img_dir = os.path.join(root,'test','ours_10000','renders')

    train_mask_dir = os.path.join(root,'train','ours_10000','masks')
    test_mask_dir = os.path.join(root,'test','ours_10000','masks')

    ll_img_ref = [os.path.join(train_img_dir,i) for i in os.listdir(train_img_dir)]
    ll_img_test = [os.path.join(test_img_dir,i) for i in os.listdir(test_img_dir)]
    
    ll_mask_ref = [os.path.join(train_mask_dir,i) for i in os.listdir(train_mask_dir)]
    ll_mask_test = [os.path.join(test_mask_dir,i) for i in os.listdir(test_mask_dir)]

    ll_dep_ref = [os.path.join(train_depth_dir,i) for i in os.listdir(train_depth_dir) if '.npy' in i]
    ll_dep_test = [os.path.join(test_depth_dir,i) for i in os.listdir(test_depth_dir) if '.npy' in i]

    ll_img_ref = sorted(ll_img_ref)
    ll_img_test = sorted(ll_img_test)
    ll_dep_ref = sorted(ll_dep_ref)
    ll_dep_test = sorted(ll_dep_test)

    ll_mask_ref = sorted(ll_mask_ref)
    ll_mask_test = sorted(ll_mask_test)
    return ll_img_ref, ll_img_test, ll_dep_ref, ll_dep_test, ll_mask_ref, ll_mask_test

if __name__ == '__main__':
    root = '/data/jianing/output_904/colmap_0_0'

    ll_img_ref, ll_img_test, ll_dep_ref, ll_dep_test, ll_mask_ref, ll_mask_test = generate_list(root)

    txt_dir = '/data/jianing/dlf_result/proj_0904_all/sparse/0'
    pcd_list = []
    for img, dep, mask in tqdm(zip(ll_img_test[:10], ll_dep_test[:10], ll_mask_test[:10])):
        # print(img)
        # print(dep)
        # print(mask)
        pcd = project_pipeline(txt_dir,img,dep,mask)
        # o3d.t.io.write_point_cloud("test.pcd", pcd)
        # break
        pcd_list.append(pcd)
    pcd = pcd_list[0]
    for i in range(1,len(pcd_list)):
        pcd =pcd + pcd_list[i]
    o3d.t.io.write_point_cloud("test.pcd", pcd)
        
