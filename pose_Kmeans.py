from kmeans import KMeans
import math
import myutils.file_utils as file_utils, myutils.pose_utils as pose_utils
import numpy as np
import cv2
import datetime
from tqdm import tqdm
import time

INPUT_JSON = "../datas/filtered_photo_data_0622.json"
IMG_DIR = "../datas/all_images/"

K = 60
EPOCHS = 1
CLUSTER_RESULT_OUTPUT = f"../datas/results/pose_cluster_kmeans_result_{datetime.datetime.today().strftime('%m%d%H%M')}.json"

start_time = time.time()
data_json = file_utils.read_json(INPUT_JSON)

featuremaps = pose_utils.get_flattened_pose_repr(data_json)


def mean(samples):
    return (np.mean(samples, axis = (0)))

best_kmeans = None
best_kmeans_distance = None

for epoch in tqdm(range(EPOCHS)):

    kmeans = KMeans(K, pose_utils.distance, mean, epoch * 42)
    kmeans.fit(np.array(featuremaps))

    result = kmeans.predict(np.array(featuremaps))
    distances = kmeans.get_average_distance(np.array(featuremaps), result)
    
    if best_kmeans == None or best_kmeans_distance > distances:
        best_kmeans = kmeans
        best_kmeans_distance = distances

kmeans = best_kmeans
distances = best_kmeans_distance

groups = {}
for i, label in enumerate(result):
    if label in groups:
        groups[int(label)].append(data_json[i]['name'])
    else:
        groups[int(label)] = [data_json[i]['name']]

save_data = {}
save_data['duration'] = time.time() - start_time
save_data['params'] = {'k' : K, 'seed' : best_kmeans.random_seed}
save_data['groups'] = groups
save_data['distances'] = distances

file_utils.save_as_json(save_data, CLUSTER_RESULT_OUTPUT)

print(f"Duration : {save_data['duration']}s")



