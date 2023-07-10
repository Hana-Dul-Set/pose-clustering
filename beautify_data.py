import sklearn.cluster
import math
import myutils.file_utils as file_utils, myutils.pose_utils as pose_utils
import myutils.image_utils as image_utils
import numpy as np
import cv2
import datetime
from tqdm import tqdm
import pandas as pd

CLUSTER_JSON = "../datas/pose_cluster_kmeans_result_06261819.json"
POSE_JSON = "../datas/filtered_photo_data_0622.json"
IMG_DIR = "../datas/all_images/"
OUTPUT_PATH = "result.csv"
DO_RENDER = True

cluster_data = file_utils.read_json(CLUSTER_JSON)

data_rows = []
for key in cluster_data['groups']:
    for name in cluster_data['groups'][key]:
        data_rows.append({'name':name, 'pose_id':key})

df = pd.DataFrame(data_rows)
df.to_csv(OUTPUT_PATH, index = False, header = ['name', 'pose_id'])
print("Done!")

print(cluster_data['params'])