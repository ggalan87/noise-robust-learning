from pathlib import Path

import numpy as np
import torch
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils.inference import FaissKNN

from features_storage import FeaturesStorage
from torchreid.metrics.rank import evaluate_rank
from torchreid.metrics.distance import compute_distance_matrix

from common_utils.etc import measure_time

from tqdm import tqdm

from torch_metric_learning.metrics.rank_cylib.rank_cy import eval_from_distmat, eval_from_sorted_indices

class CustomCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_1(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 1,
                                                  False, False, torch.eq)

    # def calculate_fancy_mutual_info(self, query_labels, cluster_labels, **kwargs):
    #     return 0
    #
    # def requires_clustering(self):
    #     return super().requires_clustering() + ["fancy_mutual_info"]

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_1"]


def load_data(cached_path):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    ((raw_training_feats, raw_testing_feats),
     (raw_training_labels, raw_testing_labels),
     (raw_training_cams, raw_testing_cams)) = fs.raw_features()

    return (raw_training_feats, raw_training_labels, raw_training_cams,
            raw_testing_feats, raw_testing_labels, raw_testing_cams)


def approach_torchreid(query_data, gallery_data):
    query_embeddings, query_labels, query_cams = query_data
    gallery_embeddings, gallery_labels, gallery_cams = gallery_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print('Computing distance matrix on feats...', gallery_embeddings.shape, query_embeddings.shape)
    # dists_np = compute_distance_matrix(query_embeddings.to(device).half(), gallery_embeddings.to(device).half()).detach().cpu().numpy()
    dists_np = compute_distance_matrix(query_embeddings.to(device), gallery_embeddings.to(device)).detach().cpu().numpy()

    query_labels_np = query_labels.detach().cpu().numpy().ravel()
    gallery_labels_np = gallery_labels.detach().cpu().numpy().ravel()
    query_cams_np = query_cams.detach().cpu().numpy().ravel()
    gallery_cams_np = gallery_cams.detach().cpu().numpy().ravel()

    # print('Running evaluation...')
    all_cmc, mAP = eval_from_distmat(dists_np,
                                 query_labels_np,
                                 gallery_labels_np,
                                 q_camids=query_cams_np,
                                 g_camids=gallery_cams_np,
                                 max_rank=50)

    rank_1 = all_cmc[0]
    print(rank_1)


def approach_pml(query_data, gallery_data):
    query_embeddings, query_labels, query_cams = query_data
    gallery_embeddings, gallery_labels, gallery_cams = gallery_data

    #calculator = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",), k=1)
    calculator = CustomCalculator(include=("precision_at_1",), k=1)

    accuracies = calculator.get_accuracy(
        query=query_embeddings, query_labels=query_labels,
        reference=gallery_embeddings, reference_labels=gallery_labels
    )
    acc = accuracies["precision_at_1"]
    print(acc)


def compute_cmc_numpy(q_pids, g_pids, q_camids, g_camids, indices, matches, max_rank=50):
    # compute cmc curve for each query
    all_cmc = []
    num_valid_q = 0.  # number of valid query

    num_q = len(q_pids)

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc


def compute_cmc_torch(q_pids, g_pids, q_camids, g_camids, indices, matches, max_rank=50):
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    max_rank = 50

    num_q = len(q_pids)

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        if not torch.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum(dim=0)
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = torch.stack(all_cmc).to(torch.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc


def approach_my(query_data, gallery_data):
    query_embeddings, query_labels, query_cams = query_data
    gallery_embeddings, gallery_labels, gallery_cams = gallery_data

    # knn_func = FaissKNN(reset_before=False, reset_after=False)
    knn_func = FaissKNN(reset_before=False, reset_after=True)
    knn_func.train(gallery_embeddings)
    distances, indices = knn_func(query_embeddings, len(gallery_embeddings))

    query_labels_np = query_labels.detach().cpu().numpy().ravel()
    gallery_labels_np = gallery_labels.detach().cpu().numpy().ravel()
    query_cams_np = query_cams.detach().cpu().numpy().ravel()
    gallery_cams_np = gallery_cams.detach().cpu().numpy().ravel()
    indices_np = indices.numpy()

    # print('Running evaluation...')
    all_cmc, mAP = eval_from_sorted_indices(indices_np,
                                 query_labels_np,
                                 gallery_labels_np,
                                 q_camids=query_cams_np,
                                 g_camids=gallery_cams_np,
                                 max_rank=50,
                                 only_cmc=True)

    acc = all_cmc[0]
    print(acc)


def compute_rank1(cached_path):
    _, _, _, test_embeddings, test_labels, test_cams = load_data(cached_path)

    gallery_embeddings, query_embeddings = test_embeddings
    gallery_labels, query_labels = test_labels
    gallery_cams, query_cams = test_cams

    gallery_data = (gallery_embeddings, gallery_labels, gallery_cams)
    query_data = (query_embeddings, query_labels, query_cams)

    eval_kwargs = \
        {
            'query_data': query_data,
            'gallery_data': gallery_data
        }
    measure_time(message='dummy', fun=approach_pml, **eval_kwargs)
    # measure_time(message='PML', fun=approach_pml, **eval_kwargs)
    measure_time(message='torchreid', fun=approach_torchreid, **eval_kwargs)
    measure_time(message='my', fun=approach_my, **eval_kwargs)


logs_root = Path('/media/amidemo/Data/object_classifier_data/logs/lightning_logs')
dm_name = 'market1501'
model_class_name = 'LitSolider'
v = 1
features_folder_name = 'features'
e = 38
run_path = logs_root / f'{dm_name}_{model_class_name}/version_{v}'
cached_path = run_path / features_folder_name / f'features_epoch-{e}.pt'
compute_rank1(cached_path)
