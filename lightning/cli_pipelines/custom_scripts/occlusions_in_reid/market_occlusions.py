from pathlib import Path
import numpy as np
import cv2
import pickle
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
from tqdm import tqdm

market_dataset_root = Path('/data/datasets/market1501/Market-1501-v15.09.15')

market_part = market_dataset_root / 'bounding_box_test'

estimator = BodyPoseEstimator(pretrained=True)

n_occluded = 0
n_non_occluded = 0

# image_name -> true/false
images_are_occluded = {}

for image_path in tqdm(list(market_part.iterdir())):
    if image_path.suffix != '.jpg':
        continue

    image_src = cv2.imread(str(image_path), flags=cv2.IMREAD_COLOR)

    keypoints = estimator(image_src)

    image_is_occluded = True

    if len(keypoints) == 1:
        missing_keypoints = len(keypoints[0][(keypoints[0] == [0, 0, 0]).all(axis=1)])
        if missing_keypoints < 6:
            image_is_occluded = False

    if image_is_occluded:
        n_occluded += 1
    else:
        n_non_occluded += 1

    images_are_occluded[image_path.name] = image_is_occluded


print(f'occluded: {n_occluded}, non-occluded: {n_non_occluded}')

# image_dst = draw_body_connections(image_src, keypoints, thickness=4, alpha=0.7)
# image_dst = draw_keypoints(image_dst, keypoints, radius=5, alpha=0.8)

pickle.dump(images_are_occluded, open('market1501_gallery_missing_lt6.pkl', 'wb'))
