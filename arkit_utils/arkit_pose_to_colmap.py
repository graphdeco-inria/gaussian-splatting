import argparse
import numpy as np
import os
from pyquaternion import Quaternion
from hloc.utils.read_write_model import Image, write_images_text

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def arkit_pose_to_colmap(dataset_base) :
    images = {}
    with open(dataset_base + "/sparse/0/images.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])

                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                c2w = np.zeros((4, 4))
                c2w[:3, :3] = qvec2rotmat(qvec)
                c2w[:3, 3] = tvec
                c2w[3, 3] = 1.0
                c2w_cv = convert_pose(c2w)

                # transform to z-up world coordinate for better visulazation
                c2w_cv = np.array([[1, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]]) @ c2w_cv
                w2c_cv = np.linalg.inv(c2w_cv)
                R = w2c_cv[:3, :3]
                q = Quaternion(matrix=R, atol=1e-06)
                qvec = np.array([q.w, q.x, q.y, q.z])
                tvec = w2c_cv[:3, -1]
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
                
    write_images_text(images=images, path=dataset_base+"/post/sparse/online/images.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ARkit pose using hloc and COLMAP")
    parser.add_argument("--input_database_path", type=str)

    args = parser.parse_args()
    input_database_path = args.input_database_path
    arkit_pose_to_colmap(input_database_path)