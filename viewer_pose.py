import myutils.file_utils as file_utils

import cv2
import numpy as np


POSE_JSON = "../sandbox/datas/filtered_photo_data_0622.json"
WINDOW_NAME = "Pose test"

pose_data = file_utils.read_json(POSE_JSON)

IMG_DIR = "../sandbox/datas/all_images/"
IMAGE_NAME = "f_30559546753-145610464@N06.jpg"

def render_keypoints(cvimage, keypoints, withids):
    for i, point in enumerate(keypoints[:26]):
        position =  (int(point[0] * cvimage.shape[1]), int(point[1] * cvimage.shape[0]))
        cv2.circle(cvimage, position, 4, keypoint_colors[i], -1)
        if withids:
            cv2.putText(cvimage, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,  keypoint_colors[i], 2, cv2.LINE_AA)
    return cvimage
