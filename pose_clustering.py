import sklearn.cluster
import math
import file_utils, pose_utils
import numpy as np
import cv2
import datetime

INPUT_JSON = "../datas/filtered_photo_data_0622.json"
IMG_DIR = "../datas/all_images/"


CLUSTER_RESULT_OUTPUT = f"../datas/pose_cluster_dbscan_result_{datetime.datetime.today().strftime('%m%d%H%M')}.json"
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

eps = 0.03
min_samples = 3

print("Start DBSCAN")
dbscan = sklearn.cluster.DBSCAN(eps= eps
                                , min_samples = min_samples, metric=distance)
result = dbscan.fit_predict(np.array(featuremaps))

print("DBSCAN done")
labels = dbscan.labels_

groups = {}

for i,x in enumerate(result):
    if x in groups:
        groups[int(x)].append(data_json[i]['name'])
    else:
        groups[int(x)] = [data_json[i]['name']]

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

save_data = {}
save_data['params'] = {'eps' : eps, 'min_samples' : min_samples}
save_data['n_clusters'] = n_clusters
save_data['n_noise'] = n_noise
save_data['groups'] = groups

file_utils.save_as_json(save_data, CLUSTER_RESULT_OUTPUT)


print("Estimated number of clusters: %d" % n_clusters)
print("Estimated number of noise points: %d" % n_noise)


winsize = (200,200)
MAX_IMG_COUNT = 6
for key in groups:
    images = []
    for name in groups[key][:MAX_IMG_COUNT]:
        img = cv2.resize(cv2.imread(IMG_DIR + name), dsize = winsize)
        images.append(img)
    
    result_image = cv2.hconcat(images)
    cv2.imshow(str(key), result_image)
    cv2.waitKey(0)

