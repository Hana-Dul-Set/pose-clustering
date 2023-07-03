import utils.file_utils as file_utils
import utils.pose_utils as pose_utils

import numpy as np
import cv2


INPUT_JSON = "../sandbox/datas/pose_cluster_kmeans_result_06261819.json"
REFERENCE_JSON = "../sandbox/datas/filtered_photo_data_0622.json"
IMG_DIR = "../sandbox/datas/all_images/"

reference_json = file_utils.read_json(REFERENCE_JSON)
data_json = file_utils.read_json(INPUT_JSON)
p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
           (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
           (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot

parent_pairs = [(0, 0), (1, 0), (2, 0), (17, 0), (18, 0), (3, 1), (4, 2), (5, 18), (6, 18), (19, 18), (11, 19), (12, 19), (7, 5), (8, 6), (9, 7), (10, 8), (13, 11), (14, 12), (15, 13), (16, 14), (24, 15), (25, 16), (20, 24), (22, 24), (21, 25), (23, 25)]


selected_names = []
selected_names.append(data_json['groups']['0'][8])
selected_names.append(data_json['groups']['0'][12])
selected_names.append(data_json['groups']['0'][13])

selected_data = []
for k in reference_json:
    if k['name'] in selected_names:
        selected_data.append(k)


for data in selected_data:
    image = cv2.imread(IMG_DIR + data['name'])
    size = data['size']

    pose = data['keypoints'][0]
    repr = pose_utils.keypoints2representation(pose, size)
    

    winsize = (640,640)

    pose_image = cv2.resize(image, dsize = winsize)
    for i, point in enumerate(pose[:26]):
        position =  (int(point[0]*winsize[0]), int(point[1]*winsize[1]))
        cv2.circle(pose_image, position, 3, p_color[i], -1)
        cv2.putText(pose_image, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,  p_color[i], 2, cv2.LINE_AA)

    norm_image = np.zeros((winsize[0], winsize[1], 3), dtype = np.uint8)
    positions = [[0,0] for x in range(26)]
    positions[0] = repr['nose_pos']
    for pair in parent_pairs:
        i = pair[0]
        point = repr['pose'][i]
        x = point[0] + positions[pair[1]][0]
        y = point[1] + positions[pair[1]][1]
        positions[i] = [x,y]
    pose_frame = (300,300)
    for i in range(26):
        position = (int(positions[i][0]*pose_frame[0]) + pose_frame[0]//2, int(positions[i][1]*pose_frame[1]) + pose_frame[0]//2)
        cv2.circle(norm_image, position, 3,  p_color[i], -1)
        cv2.putText(norm_image, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color[i], 1, cv2.LINE_AA)


    result_image = cv2.hconcat([pose_image, norm_image])
    cv2.imshow(data['name'], result_image)
    cv2.imwrite('images/'+data['name'], result_image)
    cv2.waitKey(0)
    