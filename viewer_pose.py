import myutils.file_utils as file_utils
import myutils.pose_utils as pose_utils
import myutils.image_utils as image_utils

import cv2
import numpy as np


POSE_JSON = "../datas/filtered_photo_data_0622.json"
WINDOW_NAME = "Pose test"

pose_data = file_utils.read_json(POSE_JSON)

IMG_DIR = "../datas/all_images/"
IMAGE_NAME = "f_30559546753-145610464@N06.jpg"

WINDOW_SIZE = (800,800)

def render_keypoints(cvimage, keypoints, border_size):
    left, top, right, bottom = border_size
    width = cvimage.shape[1] - right - left
    height = cvimage.shape[0] - bottom - top

    for i, point in enumerate(keypoints[:26]):
        position =  (int(point[0] * width + left), int(point[1] * height + top))
        cv2.circle(cvimage, position, 4, pose_utils.keypoint_colors[i], -1)
        cv2.putText(cvimage, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,  pose_utils.keypoint_colors[i], 2, cv2.LINE_AA)
    return cvimage

image = cv2.imread(IMG_DIR + IMAGE_NAME)
border = image_utils.predict_image_border_after_resize(image, WINDOW_SIZE)
print(border)
image = image_utils.resize_with_black_borders(image, WINDOW_SIZE)
keypoints = pose_utils.get_data(IMAGE_NAME, pose_data)['keypoints'][0]

image =  render_keypoints(image, keypoints, border)

cv2.imshow(WINDOW_NAME, image)
cv2.waitKey(0)
    
cv2.destroyAllWindows()