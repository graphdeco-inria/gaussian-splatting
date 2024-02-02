import numpy as np


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
