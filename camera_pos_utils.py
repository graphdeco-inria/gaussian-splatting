"""
Adapted from https://antimatter15.com/splat/

"""
import numpy as np
import math

M = np.eye(4, dtype=np.float64)
M[1, 1] = -1
M[2, 2] = -1

M_inv = np.linalg.inv(M)

def compose_44(r, t):
    return np.vstack((np.hstack((np.reshape(r, (3, 3)), np.reshape(t, (3, 1)))), np.array([0, 0, 0, 1])))


def decompose_44(a):
    return a[:3, :3], a[:3, 3]

def get_view_matrix(a):
    R, t = decompose_44(a)
    R = R.flatten()
    t = t

    cam_to_world = np.array([
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [-t[0]*R[0] - t[1]*R[3] - t[2]*R[6],
         -t[0]*R[1] - t[1]*R[4] - t[2]*R[7],
         -t[0]*R[2] - t[1]*R[5] - t[2]*R[8], 1]
    ])

    return cam_to_world

def rotate4(a, rad, x, y, z):

    if x == 1:
        # Define rotation matrices for intrinsic rotations
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


    pose = rotation_x @ rotation_y @ rotation_z @ a
    return pose


def rotate45(a, rad, x, y, z):
    r_mat, t_vec = decompose_44(a)
    a = (get_view_matrix(a))
    a = np.linalg.inv(a).flatten()

    # Normalize the axis
    length = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= length
    y /= length
    z /= length

    # Calculate rotation matrix components
    s = math.sin(rad)
    c = math.cos(rad)
    t = 1 - c

    b00 = x * x * t + c
    b01 = y * x * t + z * s
    b02 = z * x * t - y * s
    b10 = x * y * t - z * s
    b11 = y * y * t + c
    b12 = z * y * t + x * s
    b20 = x * z * t + y * s
    b21 = y * z * t - x * s
    b22 = z * z * t + c

    # Apply rotation to the 4x4 matrix
    result = [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        *a[12:]
    ]
    pose = np.array(result).reshape(4, 4)
    R, t = pose[:3, :3], pose[3, :3]

    return np.transpose(pose)

def translate4(a, x, y, z):
    """
    Translates camera in each axis

    @param a: current 4x4 camera
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

    return np.dot(translation_matrix, a)

if __name__ == '__main__':
    R_mat = np.array([[-0.8210050288356835, -0.17857461458472693, 0.5422746994397166],
                      [0.1249652793099283, 0.8705835196074918, 0.47588655618206266],
                      [-0.5570766747885881, 0.45847076505896, -0.6924378210299766]])
    T_vec = np.array([-1.8636133065164748, 2.1165406815192687, 3.141789771805336])
    pose = compose_44(R_mat, T_vec)
    print(pose)
    #print(np.linalg.inv(get_view_matrix(pose)))

    pose = rotate4(pose, np.radians(1), 1, 0, 0)
    print(pose)
    #pose[:3, 3] = T_vec
    #print(pose)