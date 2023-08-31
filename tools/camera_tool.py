import numpy as np

def load_params_from_file(filename):
    
    return eye, target, up

def look_at_to_extrinsic(eye, target, up):
    # Calculate the forward, right, and up vectors.
    forward = np.array(target) - np.array(eye)
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)

    # Create the transformation matrix.
    view_matrix = np.identity(4)
    view_matrix[0, :-1] = right
    view_matrix[1, :-1] = new_up
    view_matrix[2, :-1] = -forward
    view_matrix[:3, 3] = -eye
    
    return view_matrix

# Example usage:
eye = [0.0, 0.0, 3.0]     # Camera position
target = [0.0, 0.0, 0.0]  # Point the camera looks at
up = [0.0, 1.0, 0.0]      # Up vector

extrinsic_matrix = look_at_to_extrinsic(eye, target, up)
print("Camera Extrinsic Matrix:")
print(extrinsic_matrix)
