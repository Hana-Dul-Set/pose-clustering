import sklearn.cluster
import math
import myutils.file_utils as file_utils, myutils.pose_utils as pose_utils
import numpy as np
import cv2
import datetime
import time

INPUT_JSON = "../datas/filtered_photo_data_0622.json"
IMG_DIR = "../datas/all_images/"


CLUSTER_RESULT_OUTPUT = f"../datas/pose_cluster_dbscan_result_{datetime.datetime.today().strftime('%m%d%H%M')}.json"
data_json = file_utils.read_json(INPUT_JSON)

start_time = time.time()

eps = 0.9
min_samples = 3

print("Start DBSCAN")
dbscan = sklearn.cluster.DBSCAN(eps= eps
                                , min_samples = min_samples, metric= pose_utils.distance)
result = dbscan.fit_predict(pose_utils.get_flattened_pose_repr(data_json))

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
save_data['duration'] = start_time - time.time()
save_data['params'] = {'eps' : eps, 'min_samples' : min_samples}
save_data['n_clusters'] = n_clusters
save_data['n_noise'] = n_noise
save_data['groups'] = groups

file_utils.save_as_json(save_data, CLUSTER_RESULT_OUTPUT)

print("Duration : " + str(save_data['duration']))
print("Estimated number of clusters: %d" % n_clusters)
print("Estimated number of noise points: %d" % n_noise)


