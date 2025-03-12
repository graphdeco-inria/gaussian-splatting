from colmap.scripts.python.read_write_model import *

def get_colmap_data(dataset_path):
    """
    Load COLMAP data from the given dataset path.
    
    Args:
        dataset_path (str): Path to the dataset directory containing images, sparse/0, and dense/0 folders.
    
    Returns:
        images: colmap image infos
        points3D: colmap 3D points
        cameras: colmap camera infos
    """
    images = read_images_binary(os.path.join(dataset_path, 'images.bin'))
    points3D = read_points3D_binary(os.path.join(dataset_path, 'points3D.bin'))
    cameras = read_cameras_binary(os.path.join(dataset_path, 'cameras.bin'))
    return images, points3D, cameras

def quaternion_rotation_matrix(qw, qx, qy, qz):
    """
    Convert a quaternion to a rotation matrix.
    colmap uses wxyz order for quaternions.
    """
    # First row of the rotation matrix
    r00 = 2 * (qw * qw + qx * qx) - 1
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qw * qy + qx * qz)
     
    # Second row of the rotation matrix
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 2 * (qw * qw + qy * qy) - 1
    r12 = 2 * (qy * qz - qw * qx)
     
    # Third row of the rotation matrix
    r20 = 2 * (qz * qx - qw * qy)
    r21 = 2 * (qz * qy + qw * qx)
    r22 = 2 * (qw * qw + qz * qz) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def compute_intrinsic_matrix(fx, fy, cx, cy, image_width, image_height):
    sx = cx/(image_width/2)
    sy = cy/(image_height/2)
    intrinsic_matrix = np.array([[fx/sx, 0, cx/sx], [0, fy/sy, cy/sy], [0, 0, 1]])
    return intrinsic_matrix

def compute_intrinsics(colmap_cameras, image_width, image_height):
    intrinsics = {}
    for cam_key in colmap_cameras.keys():
        intrinsic_parameters = colmap_cameras[cam_key].params
        assert colmap_cameras[cam_key].model == 'PINHOLE'
        intrinsic = compute_intrinsic_matrix(intrinsic_parameters[0], 
                                             intrinsic_parameters[1], 
                                             intrinsic_parameters[2], 
                                             intrinsic_parameters[3], 
                                             image_width, 
                                             image_height)
        intrinsics[cam_key] = intrinsic
    return intrinsics

def compute_extrinsics(colmap_images):
    rotations = {}
    translations = {}
    for image_key in colmap_images.keys():
        rotation = quaternion_rotation_matrix(colmap_images[image_key].qvec[0], 
                                              colmap_images[image_key].qvec[1], 
                                              colmap_images[image_key].qvec[2], 
                                              colmap_images[image_key].qvec[3])
        translation = colmap_images[image_key].tvec
        rotations[image_key] = rotation
        translations[image_key] = translation
    return rotations, translations