import sklearn.cluster
import math
import myutils.file_utils as file_utils, myutils.pose_utils as pose_utils
import myutils.image_utils as image_utils
import numpy as np
import cv2
import datetime
from tqdm import tqdm
import random 

CLUSTER_JSON = "../datas/pose_cluster_kmeans_result_07140354.json"
POSE_JSON = "../datas/filtered_photo_data_0622.json"
IMG_DIR = "../datas/all_images/"
OUTPUT_PATH = "result.jpg"
DO_RENDER = True

image_size = (200,200)
MAX_IMG_COUNT = 10


def get_data(image_name):
    return next(x for x in pose_data if x['name'] == image_name)

cluster_data = file_utils.read_json(CLUSTER_JSON)
pose_data = file_utils.read_json(POSE_JSON)

print("[Params]\n", cluster_data['params'])

counts = {}
for key in cluster_data['groups']:
    counts[key] = len(cluster_data['groups'][key])

print("[Counts of each group]\n", counts)
counts = list(counts.values())
print("Average:",str(round(sum(counts)/len(counts))),"| Min:",min(counts),"| Max:",max(counts))

window = []
if DO_RENDER:
    for key in tqdm(cluster_data['groups'], desc="Generating images"):
        group = cluster_data['groups'][key]
        images = []

        #get average repr of all poses in group
        reprs = []
        for name in cluster_data['groups'][key]:
            data = get_data(name)
            reprs.append(pose_utils.keypoints2representation(data['keypoints'][0], data['size']))
        average_repr = pose_utils.get_average_repr(reprs)
        pose_img = image_utils.black_image(image_size)
        pose_img = pose_utils.render_representation(pose_img, average_repr, color = None, withids=True, lines=1)
        images.append(pose_img)
        for name in cluster_data['groups'][key][:MAX_IMG_COUNT]:
            img = cv2.resize(cv2.imread(IMG_DIR + name), dsize = image_size)
            images.append(img)
        
        image_row = cv2.hconcat(images)

        if MAX_IMG_COUNT > len(images)-1:
            black = np.zeros((image_size[0], image_size[1], 3), dtype = np.uint8)
            for i in range(MAX_IMG_COUNT - (len(images)-1)):
                image_row = cv2.hconcat([image_row, black])
        cv2.putText(image_row, str(key), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        window.append(image_row)
    result_image = cv2.vconcat(window)
    cv2.imwrite(OUTPUT_PATH, result_image)
    print("Saved image in", OUTPUT_PATH)
while True:
    command = input('enter] ClusterID ImageIndex >>')
    if command == 'q':
        break
    else:
        try:
            cluster, index = command.split(' ')
            index = int(index)
            if int(cluster) >= len(cluster_data['groups']):
                print(f"Cluster id is bigger than max({len(cluster_data['groups'])})")
                continue
            if index > len(cluster_data['groups'][cluster]):
                print(f"Image index is bigger than max({len(cluster_data['groups'][cluster])})")
                continue
        except:
            print("Enter again")
            continue
        print(f"Cluster {cluster}'s {index}th image : {cluster_data['groups'][cluster][index]}")
