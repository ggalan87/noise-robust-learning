import torch
from pytorch_metric_learning import distances
from torch_metric_learning.distances.algorithms.k_reciprocal_distance import k_reciprocal_distance_torch_np


class KReciprocalLpDistance(distances.LpDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_mat(self, query_emb, ref_emb=None):
        distance_mat = super().compute_mat(query_emb, ref_emb)
        refined_distance_mat = k_reciprocal_distance_torch_np(query_embeddings=None,
                                                              gallery_embeddings=None,
                                                              distance_matrix=distance_mat)

        return refined_distance_mat
