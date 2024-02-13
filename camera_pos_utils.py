import numpy as np
from scipy.spatial.transform import Rotation


class ImagesMeta:
    def __init__(self, file):
        self.img_id = []
        self.files = []
        self.t_vec = []
        self.q_vec = []

        with open(file, 'r') as f:
            for count, line in enumerate(f, start=0):
                if count < 4:
                    pass
                else:
                    if count % 2 == 0:
                        str_parsed = line.split()
                        self.img_id.append(str_parsed[0])
                        self.files.append(str_parsed[9])
                        self.t_vec.append(str_parsed[5:8])
                        self.q_vec.append(str_parsed[1:5])

        self.q_vec = np.array(self.q_vec, dtype=np.float32)
        self.t_vec = np.array(self.t_vec, dtype=np.float32)

    def get_closest_n(self, pose, n=3):
        """
        :param pose: 4x4 extrinsic matrix
        :param n: number of closest images to provide
        :return: list of image filenames
        """
        R, t = decompose_44(pose)
        R = Rotation.from_matrix(R)
        R = R.as_quat()[[3, 0, 1, 2]]  # Change from x,y,z,w to w,x,y,z

        # First filter cameras by translation
        t_dist = np.linalg.norm(self.t_vec - t, axis=1)

        lowest_t_idx = np.argsort(t_dist)[0:(2 * n)]  # Find the closest 2*n idxs
        filtered_files = [self.files[i] for i in lowest_t_idx.tolist()]

        # Then rank by camera translation
        filtered_q_vec = self.q_vec[lowest_t_idx]
        q_dist = np.linalg.norm(filtered_q_vec - R, axis=1)
        lowest_q_idx = np.argsort(q_dist)[0:n]
        q_filtered_files = [filtered_files[i] for i in lowest_q_idx.tolist()]

        return q_filtered_files

    def get_pose_by_filename(self, filename):
        idx = self.files.index(filename)
        return Rotation.from_quat(self.q_vec[idx][[1, 2, 3, 0]]).as_matrix(), self.t_vec[idx]


def compose_44(r, t):
    return np.vstack((np.hstack((np.reshape(r, (3, 3)), np.reshape(t, (3, 1)))), np.array([0, 0, 0, 1])))


def decompose_44(a):
    return a[:3, :3], a[:3, 3]


def rotate4(rad, x, y, z):
    """
    rotate4 rotates the camera around the specified axis

    @param rad: rotation angle in radians
    @param x: rotate around the camera's x-axis (pitch)
    @param y: rotate around the camera's y-axis (yaw)
    @param z: rotate around the camera's z-axis (roll)

    returns 4x4 rotation matrix (MUST BE FLOAT32)
    """

    if x == 1:
        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rad), -np.sin(rad), 0],
            [0, np.sin(rad), np.cos(rad), 0],
            [0, 0, 0, 1]
        ])
    else:
        rotation_x = np.eye(4)

    if y == 1:
        rotation_y = np.array([
            [np.cos(rad), 0, np.sin(rad), 0],
            [0, 1, 0, 0],
            [-np.sin(rad), 0, np.cos(rad), 0],
            [0, 0, 0, 1]
        ])
    else:
        rotation_y = np.eye(4)

    if z == 1:
        rotation_z = np.array([
            [np.cos(rad), -np.sin(rad), 0, 0],
            [np.sin(rad), np.cos(rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        rotation_z = np.eye(4)

    C2C_Rot = rotation_x @ rotation_y @ rotation_z
    return C2C_Rot.astype(np.float32)


def translate4(x, y, z):
    """
    Translates camera in each axis

    @param x: how much to translate in the x-axis
    @param y: how much to translate in the y-axis
    @param z: how much to translate in the z-axis

    Sample Usage:
    matrix = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    translated_matrix = translate4(matrix, 1, 1.25, 0)
    """
    translation_matrix = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    return translation_matrix.astype(np.float32)


if __name__ == '__main__':
    R_mat = np.array([[-0.70811329, -0.21124761, 0.67375813],
                      [0.16577646, 0.87778949, 0.4494483],
                      [-0.68636268, 0.42995355, -0.58655453]])
    T_vec = np.array([-0.32326042, -3.65895232, 2.27446875])
    init_pose = compose_44(R_mat, T_vec)
    imagelocs = ImagesMeta("/home/cviss/PycharmProjects/GS_Stream/data/UW_tower/sparse/0/images.txt")
    print(imagelocs.get_closest_n(init_pose))
