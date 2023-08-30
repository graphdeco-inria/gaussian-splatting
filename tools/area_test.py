import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import sys
from tqdm import tqdm
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
    left = [f'cam_{i}.png' for i in left]
    right = [f'cam_{i}.png' for i in right]

    d = sys.argv[1]
    dd_train = d + "/train/ours_10000/renders"
    dd_test = d + "/test/ours_10000/renders"
    
    
    ll_all = [os.path.join(dd_train,i)for i in os.listdir(dd_train)] + [os.path.join(dd_test,i)for i in os.listdir(dd_test)]
    
    all_area_list = []
    area_all = 0
    for ii in tqdm(ll_all):
        path = ii
        _,area = detect_red(path)
        # print(area)
        area_all += area
    print(area_all / len(ll_all))