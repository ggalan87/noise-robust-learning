# Implementation of "Graph-Based Self-Learning for Robust Person Re-identification"
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils.inference import FaissKNN
from torch_geometric.utils import get_laplacian, scatter

from torch_metric_learning.distances.k_reciprocal_lp_distance import KReciprocalLpDistance
from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection import RejectionStrategy
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import CombinedCriteria
from common_utils.etc import measure_time


class GraphBasedSelfLearning(RejectionStrategy):
    def __init__(self, epochs_frequency=2, with_relabel=True, device=torch.device('cpu')):
        super().__init__(training_samples_fraction=1.0,
                         # dummy criterion -> combined object with empty list / never called
                         rejection_criteria=CombinedCriteria([]),
                         noisy_positive_as_negative=False,
                         use_raw_probabilities=False,
                         with_relabel=with_relabel)

        self._k = 8
        self._lambda = 2

        self._epochs_frequency = epochs_frequency
        self._epoch = 0
        self._device = device

    def in_correction_epoch(self):
        return (self._epoch + 1) % self._epochs_frequency == 0

    def train(self, memory_bank: MemoryBank):
        if self.in_correction_epoch():
            # In GBSL training includes prediction, i.e. noise labels plus flip of them is done here for all dataset samples
            all_features, all_labels, all_indices, _ = memory_bank.get_data(self._training_samples_fraction)

            print('GBSL: In correction stage...')
            self._training_labels = self._compute_label_propagation(all_features, all_labels).to(
                device=all_labels.device)
            # It holds the dataset indices which correspond to the updated labels
            self._corresponding_indices = all_indices.to(device=all_labels.device)

        self._epoch += 1

    def has_trained(self):
        return self._training_labels is not None

    def _store_batch_raw_scores(self, embeddings, labels, normalize=True):
        # Nothing to do here, all job is done in train method after epoch end
        pass

    def _compute_noise_predictions(self, labels, dataset_indices=None, logits=None):
        if dataset_indices is None:
            raise AssertionError(
                'GBSL requires the dataset indices to be passed, '
                'because it keeps correct labels in random order as they come from the previous epoch dataloader')

        batch_size = len(labels)

        updated_labels = torch.zeros_like(labels).to(device=labels.device)

        # GBSL does not throw away any samples therefore we use an all true keep mask
        keep_mask = torch.ones_like(labels, dtype=torch.bool).to(device=labels.device)

        for i, dataset_index in enumerate(dataset_indices):
            item_idx = (self._corresponding_indices == dataset_index.to(self._corresponding_indices.device)).nonzero().squeeze().item()
            updated_labels[i] = self._training_labels[item_idx]

        if self.with_relabel:
            # TODO: possible refactor the following...
            # For now I just return a tuple, second part of which is the updated labels.
            self._batch_noise_predictions = (updated_labels != labels, updated_labels, keep_mask)
        else:
            self._batch_noise_predictions = updated_labels != labels

    def predict_direct_noisy(self, features, labels):
        updated_labels = self._compute_label_propagation(features, labels)

        noise_predictions = updated_labels != labels.to(device=updated_labels.device)
        return noise_predictions.to(device=labels.device)

    def _compute_affinities(self, features):
        knn_func = FaissKNN(reset_before=False, reset_after=False)

        measure_time('KNN training', knn_func.train, embeddings=features)
        _, indices = measure_time('KNN inference', knn_func, query=features, k=self._k)

        knn_func.reset()

        # Indices of nearest neighbors except self which is included in the samples
        knn_indices = indices[:, 1:]

        distance = KReciprocalLpDistance()
        refined_distance_matrix = measure_time('K-reciprocal distance', distance, query_emb=features)

        affinities_matrix = torch.zeros_like(refined_distance_matrix)

        # Paper Eq.4 - leave non neighbors with zero values
        for i in range(len(refined_distance_matrix)):
            affinities_matrix[i][knn_indices[i]] = 1. - refined_distance_matrix[i][knn_indices[i]]

        # Paper Eq. 5
        symmetric_affinities_matrix = 0.5 * (affinities_matrix + affinities_matrix.t())

        return symmetric_affinities_matrix

    @staticmethod
    def degree_matrix(input_matrix: torch.Tensor):
        input_matrix[input_matrix > 0] = 1

        result = torch.diag(torch.sum(input_matrix, dim=1))

        return result

    def _compute_label_propagation(self, features: torch.Tensor, original_labels: torch.Tensor):
        def propagation_impl(features, labels):
            affinities = self._compute_affinities(features)

            identity_matrix = torch.eye(len(affinities)).to(device=affinities.device)

            intermediate_matrix = affinities + self._lambda * identity_matrix
            deg = GraphBasedSelfLearning.degree_matrix(intermediate_matrix)

            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

            one_hot_labels = F.one_hot(labels)

            # Paper Eq. 6
            propagation_mat = deg_inv_sqrt.matmul(intermediate_matrix).matmul(deg_inv_sqrt)
            propagated_labels = propagation_mat.matmul(one_hot_labels.float())

            return torch.argmax(propagated_labels, dim=1)

        # According to the paper, label matrix is optimized with first-order message passing,
        # therefore we call the propagation function only once
        updated_labels = measure_time('Label propagation.', propagation_impl, features=features.to(device=self._device),
                                      labels=original_labels.to(device=self._device))

        return updated_labels
