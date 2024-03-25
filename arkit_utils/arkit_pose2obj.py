
import numpy as np
import math
import argparse
import os

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

def create_frustum_mesh(translation, quaternion):
    """Creates data for a camera frustum given its pose and projection parameters.

    Args:
        translation: 3D translation vector (x, y, z).
        quaternion: 4D quaternion representing camera rotation (w, x, y, z).
        fov: Field of view angle in degrees.
        aspect_ratio: Aspect ratio of the frustum's image plane.
        near: Near clipping plane distance.
        far: Far clipping plane distance.

    Returns:
        A tuple containing vertex and face data for the frustum.
    """

    # Convert quaternion to rotation matrix
    # world frame : y-up(gravity align), x-right
    # camera frame : y-up, x-right, z-point to user
    Rwc = qvec2rotmat(quaternion)
    twc = translation
    # Calculate frustum corner points in camera space
    w = 0.128/2
    h = 0.072/2
    s = 0.1/2
    top_left = [-w, -h, -s] # nagative s due to the z axis point to the user
    top_right= [w, -h, -s]
    bottom_right = [w, h, -s]
    bottom_left = [-w, h, -s]

    # Transform corner points to world space
    world_top_left = Rwc.dot(top_left) + twc
    world_top_right = Rwc.dot(top_right) + twc
    world_bottom_right = Rwc.dot(bottom_right) + twc
    world_bottom_left = Rwc.dot(bottom_left) + twc
    world_near_center = twc

    # Create vertex and face data for the frustum
    vertices = [
        "v " + " ".join([str(x) for x in world_top_left]),
        "v " + " ".join([str(x) for x in world_top_right]),
        "v " + " ".join([str(x) for x in world_bottom_right]),
        "v " + " ".join([str(x) for x in world_bottom_left]),
        "v " + " ".join([str(x) for x in world_near_center])

    ]
    faces = [
        "f 1 2 3 4",  # Front face
        "f 1 2 5",    # Left side face
        "f 1 4 5",    # Bottom side face
        "f 5 4 3",    # Right side face
        "f 2 3 5",    # Back face
    ]
    return vertices, faces

def write_multi_frustum_obj(xyzs, qxyzs, filename="multi_frustums.obj"):
    """Writes camera frustums from multiple poses into a single .obj file.

    Args:
        camera_poses: List of dictionaries with "translation" and "quaternion" keys.
        fov: Field of view angle in degrees.
        aspect_ratio: Aspect ratio of the frustum's image plane.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        filename: Output filename for the .obj file.
    """

    # obj_data = ""

    for i, pose in enumerate(xyzs):
        translation = xyzs[i]
        quaternion = qxyzs[i]
        vertices, faces = create_frustum_mesh(translation, quaternion)
        obj_data = "\n".join(vertices) + "\n"+ "\n".join(faces)
        # # Offset vertex indices based on existing vertices
        # offset = len(obj_data.split("v ")) - 1  # Number of existing vertices
        # for v in vertices:
        #     print(v)
        #     for x in v:
        #         print(x)
        # new_vertices = [" ".join([str(float(x) + offset) for x in v]) for v in vertices]
        # print(new_vertices)
        # obj_data += "v " + "\n".join(new_vertices) + "\n"

        # # Keep face indices as-is since they point to relative vertex positions
        # new_faces = ["f " + f for f in faces]
        # obj_data += "f " + "\n".join(new_faces) + "\n"

        # Write complete obj data to file
        with open(filename+str(i)+"image.obj", "w") as f:
            f.write(obj_data)

def readPose2buffer(path):
    num_frames = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_frames += 1

    print(f"num of frames : {num_frames}")
    xyzs = np.empty((num_frames, 3))
    qxyzs = np.empty((num_frames, 4))

    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 :
                elems = line.split()
                qxyz = np.array(tuple(map(float, elems[1:5])))
                xyz = np.array(tuple(map(float, elems[5:8])))
                qxyzs[count] = qxyz
                xyzs[count] = xyz
                count+=1
    
    return xyzs, qxyzs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transform ARKit pose to obj for meshLab visulization")
    parser.add_argument("--input_cameras_path", type=str)
    parser.add_argument("--output_frustum_path", type=str)
    args = parser.parse_args()
    input_cameras_path = args.input_cameras_path
    output_frustum_path = args.output_frustum_path

    if not os.path.exists(output_frustum_path):
        os.makedirs(output_frustum_path)

    xyzs, qxyzs = readPose2buffer(input_cameras_path)
    write_multi_frustum_obj(xyzs, qxyzs, output_frustum_path) 

    