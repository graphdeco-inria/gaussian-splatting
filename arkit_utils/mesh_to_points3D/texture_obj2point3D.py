
import numpy as np
import argparse

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def transformARkitRgbPCL2COLMAPpoint3DwithRgbAndZaxisUpward(input_obj_path, output_ply_path):
    num_points = 0
    with open(input_obj_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                if elems[0] == "v":
                    num_points += 1
    print(num_points)
    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))

    count = 0
    with open(input_obj_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] == "v":
                elems = line.split()
                if elems[0] == "v":
                    xyz = np.array(tuple(map(float, elems[1:4])))
                    # rotated y-up world frame to z-up world frame
                    xyz = rotx(np.pi / 2) @ xyz
                    xyzs[count] = xyz
                    rgb = np.array(tuple(map(float, elems[4:7])))
                    rgbs[count] = rgb*255
                    count+=1


    with open(output_ply_path, "w") as f:
        for i in range(num_points):
            line = str(i) + " " + str(xyzs[i][0]) + " " + str(xyzs[i][1])+ " " + str(xyzs[i][2])+ " " + str(int(rgbs[i][0])) + " " + str(int(rgbs[i][1]))+ " " + str(int(rgbs[i][2]))+ " " + str(0)
            f.write(line  + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transform ARKit texture mesh point cloud to COLMAP point3D format with RGB value and z-up coordinate")
    parser.add_argument("--input_obj_path", type=str, default="data/homee/colmap/3dgs.obj")
    parser.add_argument("--output_ply_path", type=str, default="data/homee/colmap/point3D.txt")

    args = parser.parse_args()
    input_obj_path = args.input_obj_path
    output_ply_path = args.output_ply_path

    transformARkitRgbPCL2COLMAPpoint3DwithRgbAndZaxisUpward(input_obj_path, output_ply_path)