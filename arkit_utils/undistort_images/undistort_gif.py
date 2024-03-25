import cv2
import imageio
import numpy as np
def main():
    # 讀取原始影像和 undistort 完畢的影像
    original_image = cv2.imread("data/homee/study_room_test/dis/0043.jpg")
    undistorted_image = cv2.imread("data/homee/study_room_test/images/0043.jpg")
    COLMAP_undistorted_image = cv2.imread("data/homee/study_room_colmap/images/0043.jpg")

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
    COLMAP_undistorted_image = cv2.cvtColor(COLMAP_undistorted_image, cv2.COLOR_BGR2RGB)
    height, width, channel = undistorted_image.shape
    print(f"height : {height}, width : {width}")

    text = "undistorted"
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (255, 0, 0)  # Red color in RGB format
    cv2.putText(undistorted_image, text, (100, 100), font_face, font_scale, text_color, font_thickness)
    text = "COLMAP undistorted"
    cv2.putText(COLMAP_undistorted_image, text, (100, 100), font_face, font_scale, text_color, font_thickness)

    # 設定 GIF 動畫的參數
    total_duration = 10  # 動畫播放時間

    # 建立一個空的 GIF 動畫
    with imageio.get_writer("before_and_after.gif", mode="I", duration=1000) as writer:
        for i in range(total_duration):
            writer.append_data(original_image)
            writer.append_data(undistorted_image)
            writer.append_data(COLMAP_undistorted_image)


if __name__ == "__main__":
    main()