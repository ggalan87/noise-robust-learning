from pathlib import Path
import time
import numpy as np
from scipy.spatial.distance import cdist
import torch
from pytorch_metric_learning.distances import LpDistance
from torchmetrics.classification import BinaryConfusionMatrix, BinaryF1Score

from features_storage import FeaturesStorage
from tqdm import tqdm
from common_utils.etc import measure_time
from pytorch_metric_learning.utils.inference import FaissKNN
from torch_metric_learning.distances.algorithms.k_reciprocal_distance import *
from torch_metric_learning.distances.k_reciprocal_lp_distance import KReciprocalLpDistance

from torch_metric_learning.noise_reducers.sample_rejection.gbsl import GraphBasedSelfLearning


def run_comparison():
    dm_name = 'market1501'
    model_class_name = 'LitSolider'
    v = 30
    e = 38

    run_path = \
        Path(
            f'/media/amidemo/Data/object_classifier_data/logs/lightning_logs/{dm_name}_{model_class_name}/version_{v}')

    features_folder_name = 'features'
    cached_path = run_path / features_folder_name / f'features_epoch-{e}.pt'

    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    ((raw_training_feats, _), (raw_training_labels, _), (raw_training_cams, _)) = fs.raw_features()
    raw_training_is_noisy = fs.training_feats['is_noisy']

    training_perm = torch.randperm(len(raw_training_feats))

    # refined_distance_orig = original_reranking(query_embeddings.numpy(), gallery_embeddings.numpy())
    # refined_distance_torch = measure_time('reranking torch',
    #     torch_reranking, query_features=query_embeddings, gallery_features=gallery_embeddings)

    # refined_distance_numpy = measure_time('k_reciprocal_distance_torch_np',
    #                                       k_reciprocal_distance_torch_np,
    #                                       query_embeddings=torch.cat((query_embeddings, gallery_embeddings)),
    #                                       gallery_embeddings=None)

    # my_distance_torch = measure_time('k_reciprocal_distance_torch',
    #                                  k_reciprocal_distance_torch,
    #                                  query_embeddings=query_embeddings,
    #                                  gallery_embeddings=gallery_embeddings)

    features = raw_training_feats
    labels = raw_training_labels

    gbsl_algo = GraphBasedSelfLearning()
    noise_predictions = gbsl_algo.predict_direct_noisy(features, labels)

    print(noise_predictions.device)

    n_changed_labels = torch.count_nonzero(noise_predictions)

    all_predicted_as_clean = torch.logical_not(noise_predictions)
    all_gt_clean = torch.logical_not(raw_training_is_noisy)

    confusion_matrix_metric = BinaryConfusionMatrix(normalize='true')
    confusion_matrix = confusion_matrix_metric(all_predicted_as_clean, all_gt_clean)

    f1_score_metric = BinaryF1Score()
    f1_score = f1_score_metric(all_predicted_as_clean, all_gt_clean)

    print(confusion_matrix)
    print(f1_score)

    print(f'#{n_changed_labels} out of {len(labels)} ({n_changed_labels/len(labels) * 100:.2f}%) were changed')


if __name__ == '__main__':
    run_comparison()
