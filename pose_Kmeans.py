from kmeans import KMeans
import math
import myutils.file_utils as file_utils, myutils.pose_utils as pose_utils
import numpy as np
import cv2
import datetime

INPUT_JSON = "../datas/filtered_photo_data_0622.json"
IMG_DIR = "../datas/all_images/"


CLUSTER_RESULT_OUTPUT = f"../datas/pose_cluster_kmeans_result_{datetime.datetime.today().strftime('%m%d%H%M')}.json"
data_json = file_utils.read_json(INPUT_JSON)

featuremaps = pose_utils.get_flattened_pose_repr(data_json)


def mean(samples):
    return (np.mean(samples, axis = (0)))

print("Kmeans start")

K = 50
kmeans = KMeans(K, pose_utils.distance, mean)
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



