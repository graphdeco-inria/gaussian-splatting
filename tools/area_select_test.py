import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from natsort import natsorted
def detect_red(image_path):
    # Load the image
    image = cv2.imread(image_path)

    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red = np.array([0, 130, 56])
    upper_red = np.array([10, 255, 255])

    # Create a mask for red regions
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sum = 0

    contours_filter = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            contour_sum+=area 
            contours_filter.append(contour)
        #print(f"Contour Area: {area}")

    # Draw contours on the original image
    result_image = image.copy()
    cv2.drawContours(result_image, contours_filter, -1, (0, 255, 0), 2)

    return result_image, contour_sum

if __name__ == "__main__":
    left = [11,12,10,19,17,6,15,16,9,14]
    right = [18,1,5,4,7,13,8,2,3,0]
    left = [f'cam_{i}.jpg' for i in left]
    right = [f'cam_{i}.jpg' for i in right]

    dd_train = "/home/luvision/project/gaussian-splatting/output_816_0_0/train/ours_10000/renders"
    dd_test = "/home/luvision/project/gaussian-splatting/output_816_0_0/test/ours_10000/renders"
    
    frames = {}
    
    train_list = [os.path.join(dd_train,i) for i in os.listdir(dd_train)]
    #train_list.sort()

    test_list = [os.path.join(dd_test,i) for i in os.listdir(dd_test)]
    #test_list.sort()

    all_list = train_list + test_list
    all_list.sort()

    for img in all_list:
        time = os.path.basename(img).split('_')[1]
        if time not in frames:
            frames[time] = []
            frames[time].append(img)
        else:
            frames[time].append(img)
    
    key_list = list(frames.keys())
    key_list = natsorted(key_list)

    right_area_list = []
    left_area_list = []
    for ii in key_list:
        cam_list = frames[ii]
        left_area = 0
        right_area = 0
        for cam in cam_list:
            cam_name = 'cam_' + cam.split('_')[-1]
            _, area = detect_red(cam)
            if cam_name in left:
                #print(cam_name)
                left_area += area
            else:
                right_area += area
        
        left_area_list.append(left_area)
        right_area_list.append(right_area)

    print(left_area_list)
    print(right_area_list)
    print(left_area_list.index(max(left_area_list)))
    print(right_area_list.index(max(right_area_list)))

    print(left_area_list.index(min(left_area_list)))
    print(right_area_list.index(min(right_area_list)))