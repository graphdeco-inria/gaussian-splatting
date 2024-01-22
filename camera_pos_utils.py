"""
Adapted from https://antimatter15.com/splat/

"""
import numpy as np


def compose_44(r, t):
    return np.vstack((np.hstack((np.reshape(r, (3, 3)), np.reshape(t, (3, 1)))), np.array([0, 0, 0, 1])))


def decompose_44(a):
    return a[:3, :3], a[:3, 3]


def rotate4(a, rad, x, y, z):
    """
    Rotates camera about the specified axis by radians

    @param a: current 4x4 camera extrinsic matrix
    @param rad: rotation angle in radians
    @param x: int(1) or int(0) indicating to rotate about the x-axis
    @param y: int(1) or int(0) indicating to rotate about the y-axis
    @param z: int(1) or int(0) indicating to rotation about the z-axis
    @return: new 4x4 camera extrinsic matrix

    Sample Use:
    matrix = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    Rotate 45 degrees about the x-axis
    rotated_matrix = rotate4(matrix, np.radians(45), 1, 0, 0)
    """
    len_xyz = np.linalg.norm([x, y, z])
    x /= len_xyz
    y /= len_xyz
    z /= len_xyz
    c = np.cos(rad)
    s = np.sin(rad)
    t = 1 - c

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.array([
        [x * x * t + c, y * x * t - z * s, z * x * t + y * s],
        [x * y * t + z * s, y * y * t + c, z * y * t - x * s],
        [x * z * t - y * s, y * z * t + x * s, z * z * t + c]
    ])

    return np.matmul(a, rotation_matrix)


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
