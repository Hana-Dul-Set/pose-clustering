import sklearn.cluster
import math
import file_utils, pose_utils
import numpy as np
import cv2
import datetime
import random 
INPUT_JSON = "../sandbox/datas/pose_cluster_dbscan_result_06261107.json"
IMG_DIR = "../sandbox/datas/all_images/"

data_json = file_utils.read_json(INPUT_JSON)

print("params:",data_json['params'])
print("cluster count:",data_json['n_clusters'], "noise points:", data_json['n_noise'])
winsize = (100,100)
MAX_IMG_COUNT = 16

window = []
for key in data_json['groups']:
    group = data_json['groups'][key]
    images = []
    #random.shuffle(group)
    print(key,':',len(group))
    if key=='0':
        print(group[:10])
    for name in data_json['groups'][key][:MAX_IMG_COUNT]:
        img = cv2.resize(cv2.imread(IMG_DIR + name), dsize = winsize)
        images.append(img)
    
    image_row = cv2.hconcat(images)
    if MAX_IMG_COUNT > len(images):
        black = np.zeros((winsize[0], winsize[1], 3), dtype = np.uint8)
        for i in range(MAX_IMG_COUNT - len(images)):
            image_row = cv2.hconcat([image_row, black])
    cv2.putText(image_row, str(key), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    window.append(image_row)
    
result_image = cv2.vconcat(window)
cv2.imshow(str(key), result_image)
cv2.waitKey(0)
