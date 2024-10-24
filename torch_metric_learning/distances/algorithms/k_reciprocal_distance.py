import warnings
from typing import Optional
import pytorch_metric_learning.distances.lp_distance
from tqdm import tqdm
import numpy as np
import torch
from pytorch_metric_learning.distances import LpDistance


def k_reciprocal_distance_torch_np(query_embeddings: Optional[torch.Tensor],
                                   gallery_embeddings: Optional[torch.Tensor],
                                   distance_matrix: Optional[torch.Tensor] = None,
                                   k1: int = 20,
                                   k2: int = 6,
                                   lambda_value: float = 0.3) -> torch.Tensor:
    """
    Adapted from: https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/re_ranking.py
    """

    def k_reciprocal_neighbor(initial_rank, i, k):
        forward_k_neigh_index = initial_rank[i, :k + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    if ((query_embeddings is None and distance_matrix is None) or
            (query_embeddings is not None and distance_matrix is not None)):
        raise ValueError('One of the query embeddings or distance matrix should be set')

    if distance_matrix is None:
        distance = LpDistance()
        q_q_dist = distance(query_embeddings).numpy()

        if gallery_embeddings is not None:
            q_g_dist = distance(query_embeddings, gallery_embeddings).numpy()
            g_g_dist = distance(gallery_embeddings).numpy()

            original_dist = np.concatenate(
                [np.concatenate([q_q_dist, q_g_dist], axis=1),
                 np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
                axis=0)
        else:
            original_dist = q_q_dist

        query_num = q_q_dist.shape[0]
        all_num = original_dist.shape[0]
    else:
        original_dist = distance_matrix.cpu().numpy()
        query_num = all_num = original_dist.shape[0]

    del distance_matrix

    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0)).astype(np.float16)
    computations_dtype = original_dist.dtype

    if lambda_value == 1.0:
        return torch.tensor(original_dist)

    V = np.zeros_like(original_dist).astype(computations_dtype)
    initial_rank = np.argpartition(a=original_dist, kth=range(1, k1 + 1))

    for i in tqdm(range(all_num)):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neighbor(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neighbor(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=computations_dtype)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=computations_dtype)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=computations_dtype)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist

    if query_num != all_num:
        # In the case of cross distance we should obtain the relevant tile of the large matrix
        final_dist = final_dist[:query_num, query_num:]
    else:
        # In the case of self distances there are very small negative values (probably arisen due to eps corrections)
        # that I zero them out
        final_dist[final_dist < 0] = 0

    return torch.tensor(final_dist)


def k_reciprocal_distance_torch(query_embeddings: Optional[torch.Tensor],
                                gallery_embeddings: Optional[torch.Tensor],
                                distance_matrix: Optional[torch.Tensor] = None,
                                k1: int = 20,
                                k2: int = 6,
                                lambda_value: float = 0.3) -> torch.Tensor:
    """
    Adapted from: https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/re_ranking.py to torch

    WARNING: much slower than numpy version.
    """

    def k_reciprocal_neighbor(initial_rank, i, k):
        forward_k_neigh_index = initial_rank[i, :k + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    if ((query_embeddings is None and distance_matrix is None) or
            (query_embeddings is not None and distance_matrix is not None)):
        raise ValueError('One of the query embeddings or distance matrix should be set')

    if distance_matrix is None:
        if gallery_embeddings is not None:
            feat = torch.cat((query_embeddings, gallery_embeddings))
        else:
            feat = query_embeddings

        query_num, all_num = query_embeddings.size(0), feat.size(0)
        feat = feat.view(all_num, -1)

        distance = LpDistance()
        original_dist = distance(feat)
    else:
        original_dist = distance_matrix
        query_num = all_num = original_dist.shape[0]

    warnings.warn('Torch implementation is slow. Use numpy version!')

    original_dist = (1. * original_dist / torch.max(original_dist, dim=0).values).t()
    V = torch.zeros_like(original_dist)

    initial_rank = torch.topk(original_dist, k=k1, largest=False).indices

    for i in tqdm(range(all_num)):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neighbor(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neighbor(initial_rank, candidate, int(np.around(k1 / 2)))
            # https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/11
            intersection = candidate_k_reciprocal_index[
                (candidate_k_reciprocal_index.view(1, -1) == k_reciprocal_index.view(-1, 1)).any(dim=0)]

            if len(intersection) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / torch.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = torch.zeros_like(V, dtype=torch.float32)
        for i in range(all_num):
            V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(torch.where(V[:, i] != 0)[0])

    jaccard_dist = torch.zeros_like(original_dist, dtype=torch.float32)

    for i in range(query_num):
        temp_min = torch.zeros(size=(1, all_num), dtype=torch.float32)
        indNonZero = torch.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.minimum(V[i, indNonZero[j]],
                                                                                  V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
