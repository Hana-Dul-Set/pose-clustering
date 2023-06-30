import sklearn.cluster
import math
import file_utils, pose_utils
import numpy as np
import cv2
import datetime
import random 
INPUT_JSON = "../sandbox/datas/pose_cluster_kmeans_result_06261819.json"
IMG_DIR = "../sandbox/datas/all_images/"

data_json = file_utils.read_json(INPUT_JSON)

print("params:",data_json['params'])

winsize = (150,150)
MAX_IMG_COUNT = 6
counts = {}
for key in data_json['groups']:
    print(key,":", len(data_json['groups'][key]))
    counts[key] = len(data_json['groups'][key])

counts = list(counts.values())
print(min(counts))
print(max(counts))

print()

window = []
for key in data_json['groups']:
    group = data_json['groups'][key]
    images = []
    random.shuffle(group)
    print(key,':',len(group))
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
