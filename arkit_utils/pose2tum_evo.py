import numpy as np
import struct

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

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def read_pose_txt(path):
    txt_path = path + "images.txt"
    num_frames = 0
    with open(txt_path, "r") as fid:
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
    with open(txt_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                qxyz = np.array(tuple(map(float, elems[1:5])))
                xyz = np.array(tuple(map(float, elems[5:8])))

                Twc = np.zeros((4, 4))
                Twc[:3, :3] = qvec2rotmat(qxyz)
                Twc[:3, 3] = xyz
                Twc[3, 3] = 1.0
                Twc = convert_pose(Twc)
                Twc = np.array([[1, 0, 0, 0],
                     [0, 0, -1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]]) @ Twc
                
                R = Twc[:3, :3]
                qvec = rotmat2qvec(R)
                tvec = Twc[:3, -1]

                qxyzs[count] = qxyz
                xyzs[count] = xyz
                count+=1
    
    write_2_TUM_format(num_frames, xyzs, qxyzs, path+"est_tum.txt")


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_pose_bin(path):
    num_frames = 0
    bin_path  = path + "images.bin"
    with open(bin_path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            num_frames += 1
    print(f"num of frames : {num_frames}")
    xyzs = np.empty((num_frames, 3))
    qxyzs = np.empty((num_frames, 4))

    count = 0

    with open(bin_path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            
            # COLMAP pose is in Tcw, we need Twc
            Tcw = np.zeros((4, 4))
            Tcw[:3, :3] = qvec2rotmat(qvec)
            Tcw[:3, 3] = tvec
            Tcw[3, 3] = 1.0

            Twc = np.linalg.inv(Tcw)
            R = Twc[:3, :3]
            qvec = rotmat2qvec(R)
            tvec = Twc[:3, -1]
            
            # binary won't read as increasing order
            qxyzs[image_id-1] = qvec
            xyzs[image_id-1] = tvec
            count+=1
    
    write_2_TUM_format(num_frames, xyzs, qxyzs, path+"gt_tum.txt")


def write_2_TUM_format(n, xyzs, qxyzs, path):
    '''
    tum expect pose in Twc (camera to world)
    '''
    with open(path, "w") as f:
        for i in range(n):
            line = str(i) + " " + str(xyzs[i][0]) + " " + str(xyzs[i][1])+ " " + str(xyzs[i][2])+ " " + str(qxyzs[i][1])+ " " + str(qxyzs[i][2])+ " " + str(qxyzs[i][3]) + " " + str(qxyzs[i][0])
            f.write(line  + "\n")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="transform ARKit pose to obj for meshLab visulization")
    # parser.add_argument("--input_cameras_path", type=str)
    # args = parser.parse_args()

    # input_cameras_path = args.input_cameras_path

    # read_pose_txt("data/arkit_pose/meeting_room_loop_closure/arkit_colmap2/colmap_arkit/raw/")
    read_pose_bin("data/arkit_pose/meeting_room_loop_closure/arkit_colmap/colmap_arkit/raw/colmap_ba/")

