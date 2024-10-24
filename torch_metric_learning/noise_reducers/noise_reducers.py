from abc import abstractmethod
import torch

from .memory_bank import MemoryBank
from .sample_rejection import RejectionStrategy, GraphBasedSelfLearning
from .sample_rejection.rejection_base import Inspector
from common_utils.etc import measure_time


class DefaultNoiseReducer(torch.nn.Module):
    def __init__(self, strategy: RejectionStrategy, memory_bank, warm_up_epochs: int = 0, keep_only_good_samples=False,
                 use_pretrained=False, with_inspector=False, with_final_epoch_features=False):
        """

        @param strategy: a rejection strategy, which realizes if a sample is  noisy or not
        @param memory_size: 0 means dynamic memory, above zero means static preallocated memory
        @param warm_up_epochs: number of starting epochs to mute the noise reducer
        @param keep_only_good_samples: whether to store only clean samples to memory
        @param use_pretrained: use pretrained network to initialize the noise reduction before training starts
        """
        super().__init__()

        if warm_up_epochs > 0 and use_pretrained:
            raise ValueError('You required to use pretrained network weights with warm up. This will have no effect!')

        self._strategy = strategy
        self._keep_only_good_samples = keep_only_good_samples
        self._use_pretrained = use_pretrained
        self.current_epoch = 0
        self.warm_up_epochs = warm_up_epochs

        self._memory_bank = memory_bank
        self._with_final_epoch_features = with_final_epoch_features
        self._inspector = Inspector() if with_inspector else None

    def forward(self, embeddings, labels, noisy_samples=None, dataset_indices=None, logits=None):
        """

        @param embeddings: the embeddings of th batch
        @param labels: the labels of the batch
        @param noisy_samples: ground truth noisy samples, used for debugging/evaluation
        @param dataset_indices: dataset indices of the samples
        @return:
        """
        self._strategy.current_batch_noisy_samples = noisy_samples
        return self._noise_reduction_impl(embeddings, labels, dataset_indices, logits=logits)

    def bootstrap_initial(self, embeddings, labels, dataset_indices=None):
        """ Can be called before starting training to initialize with pretrained weights"""
        print('Using vanilla features for training...')
        self._memory_bank.add_to_memory(embeddings, labels, dataset_indices=dataset_indices)
        self._strategy.train(self._memory_bank)

    def bootstrap_epoch(self, epoch=-1, embeddings=None, labels=None, dataset_indices=None):
        """
        This function should be called inside training_epoch_end() hook. In this case the epoch has already
        incremented by 1.
        TODO: Consider altering logic for epoch_start / there is no such hook in lightning (1.8.x)

        @param epoch:
        @return:
        """
        if self.current_epoch >= self.warm_up_epochs:
            if self._inspector is not None:
                self._inspector.report_and_reset(self._strategy.labels_to_indices)

            if embeddings is not None and labels is not None:
                print('Adding data to memory bank after epoch end...')
                self._memory_bank.add_to_memory(embeddings, labels, dataset_indices=dataset_indices)

            # self._strategy.train(self._memory_bank)
            measure_time('Strategy training', self._strategy.train, memory_bank=self._memory_bank)
        else:
            # TODO: since refactoring we know inside this class when to fill the memory, therefore we can omit this
            #  call
            self._memory_bank.reset_memory()

        if epoch == -1:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

    @property
    def use_pretrained(self):
        return self._use_pretrained

    @property
    def inspector(self):
        return self._inspector

    @property
    def strategy(self):
        return self._strategy

    def memory_is_dynamic(self):
        return self._memory_bank.with_dynamic_memory

    @property
    def with_epoch_features(self):
        return self._with_final_epoch_features

    def use_current_epoch_features(self):
        return True

    def has_trained(self):
        return self._strategy.has_trained()

    def _noise_reduction_impl(self, embeddings, labels, dataset_indices=None, logits=None):
        # TODO: Consider move this logic inside rejection strategy
        if self._strategy.has_trained():
            batch_noisy_predictions = self._strategy.predict_noise(embeddings, labels, dataset_indices=dataset_indices,
                                                                   logits=logits)

            # Ignore relabeling here, since rejection strategy is not aware of epochs.
            # TODO: Consider having epoch in rejection strategy, too.
            if self._strategy.with_relabel and self.current_epoch < self._strategy.relabel_starting_epoch:
                batch_noisy_predictions = batch_noisy_predictions[0]

        else:
            batch_noisy_predictions = torch.zeros_like(labels, dtype=torch.bool)

        kept_embeddings = embeddings.clone().detach()  # .to(dtype=torch.float16)
        kept_labels = labels.clone().detach()
        kept_dataset_indices = dataset_indices.clone().detach()

        do_relabel = isinstance(batch_noisy_predictions, tuple)
        # For now I use it explicitly only with relabel
        if self._keep_only_good_samples and do_relabel:
            good_samples = torch.logical_not(batch_noisy_predictions if not do_relabel else batch_noisy_predictions[0])
            kept_embeddings = kept_embeddings[good_samples]
            kept_labels = kept_labels[good_samples]
            kept_dataset_indices = kept_dataset_indices[good_samples]

        # Store current features
        self._memory_bank.add_to_memory(kept_embeddings, kept_labels, dataset_indices=kept_dataset_indices)

        # Report if needed
        if self._inspector is not None:
            self._inspector.add_batch_info(batch_labels=labels,
                                           batch_predictions_scores=self._strategy.retrieve_batch_raw_scores(),
                                           batch_predictions_noisy=batch_noisy_predictions,
                                           batch_gt_noisy=self.current_batch_noisy_samples)

        return batch_noisy_predictions


class GBSLNoiseReducer(DefaultNoiseReducer):
    def __init__(self, strategy: GraphBasedSelfLearning, memory_bank, warm_up_epochs: int = 0,
                 keep_only_good_samples=False,
                 use_pretrained=False, with_inspector=False):

        # Below I hardcode the option with_final_epoch_features because GBSL works as such
        super().__init__(strategy=strategy, memory_bank=memory_bank, warm_up_epochs=warm_up_epochs,
                         keep_only_good_samples=keep_only_good_samples, use_pretrained=use_pretrained,
                         with_inspector=with_inspector, with_final_epoch_features=True)

    def _noise_reduction_impl(self, embeddings, labels, dataset_indices=None, logits=None):
        # TODO: Consider move this logic inside rejection strategy
        if self._strategy.has_trained():
            batch_noisy_predictions = self._strategy.predict_noise(embeddings, labels, dataset_indices=dataset_indices)
        else:
            batch_noisy_predictions = torch.zeros_like(labels, dtype=torch.bool)

        # IMPORTANT: In this overriden function, store to memory has been omitted, since in GBSL features are extracted
        # in the end of the epoch and not kept in the memory bank during training.

        # Report if needed
        if self._inspector is not None:
            self._inspector.add_batch_info(batch_labels=labels,
                                           batch_predictions_scores=self._strategy.retrieve_batch_raw_scores(),
                                           batch_predictions_noisy=batch_noisy_predictions,
                                           batch_gt_noisy=self.current_batch_noisy_samples)

        return batch_noisy_predictions

    def use_current_epoch_features(self):
        return self._strategy.in_correction_epoch()
