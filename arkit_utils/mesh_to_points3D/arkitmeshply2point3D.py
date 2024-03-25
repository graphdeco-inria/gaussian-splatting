
import numpy as np
import argparse

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def transformARkitPCL2COLMAPpoint3DwithZaxisUpward(input_ply_path, output_ply_path):
    find_start_row = False
    num_points = 0
    with open(input_ply_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0:
                elems = line.split()
                if find_start_row == False and elems[0] == "end_header":
                    find_start_row = True
                    continue
                if find_start_row and len(elems)==3:
                    num_points += 1
    print(f"total num of point cloud : {num_points}")
    xyzs = np.empty((num_points, 3))
    rgbs = np.zeros((num_points, 3))

    count = 0

    find_start_row = False # reset

    with open(input_ply_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 :
                elems = line.split()
                if find_start_row == False and elems[0] == "end_header":
                    find_start_row = True
                    continue
                if find_start_row and len(elems)==3:
                    xyz = np.array(tuple(map(float, elems[0:3])))
                    # rotated y-up world frame to z-up world frame
                    xyz = rotx(np.pi / 2) @ xyz
                    xyzs[count] = xyz
                    count+=1


    with open(output_ply_path, "w") as f:
        for i in range(num_points):
            line = str(i) + " " + str(xyzs[i][0]) + " " + str(xyzs[i][1])+ " " + str(xyzs[i][2])+ " " + str(int(rgbs[i][0])) + " " + str(int(rgbs[i][1]))+ " " + str(int(rgbs[i][2]))+ " " + str(0)
            f.write(line  + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transform ARKit mesh point cloud to COLMAP point3D format with z-up coordinate")
    parser.add_argument("--input_base_path", type=str, default="data/homee/colmap")

    args = parser.parse_args()
    input_ply_path = args.input_base_path + "/sparse/ARKitmesh.ply"
    output_ply_path = args.input_base_path + "/post/sparse/online/points3D.txt"

    transformARkitPCL2COLMAPpoint3DwithZaxisUpward(input_ply_path, output_ply_path)