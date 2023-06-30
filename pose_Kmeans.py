from kmeans import KMeans
import math
import file_utils, pose_utils
import numpy as np
import cv2
import datetime

INPUT_JSON = "../sandbox/datas/filtered_photo_data_0622.json"
IMG_DIR = "../sandbox/datas/all_images/"


CLUSTER_RESULT_OUTPUT = f"../sandbox/datas/random/pose_cluster_kmeans_result_{datetime.datetime.today().strftime('%m%d%H%M')}.json"
data_json = file_utils.read_json(INPUT_JSON)

featuremaps = []

for data in data_json:
    size = data['size']
    pose = data['keypoints'][0]
    featuremaps.append(pose_utils.keypoints2representation(pose, size)['pose'].flatten())
    
def distance(A, B):
    a = A.reshape((A.size//2, 2))
    b = B.reshape((B.size//2, 2))
    return np.mean(np.linalg.norm(a - b, axis = 1))

def mean(samples):
    return (np.mean(samples, axis = (0)))

print("Kmeans start")

K = 100
kmeans = KMeans(K, distance, mean)
kmeans.fit(np.array(featuremaps))

print("Kmeans done")
result = kmeans.predict(np.array(featuremaps))

distances = kmeans.get_average_distance(np.array(featuremaps), result)
print(distances)

groups = {}
for i, label in enumerate(result):
    if label in groups:
        groups[int(label)].append(data_json[i]['name'])
    else:
        groups[int(label)] = [data_json[i]['name']]

save_data = {}
save_data['params'] = {'k' : K}
save_data['groups'] = groups
save_data['distances'] = distances

file_utils.save_as_json(save_data, CLUSTER_RESULT_OUTPUT)


winsize = (200,200)
MAX_IMG_COUNT = 10
for key in groups:
    images = []
    for name in groups[key][:MAX_IMG_COUNT]:
        img = cv2.resize(cv2.imread(IMG_DIR + name), dsize = winsize)
        images.append(img)
    
    result_image = cv2.hconcat(images)
    cv2.imshow(str(key), result_image)
    cv2.waitKey(0)

