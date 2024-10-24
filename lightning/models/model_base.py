from typing import Tuple, Union, Optional, Type
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from lightning_lite.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning import LightningModule
# from pytorch_lightning.core.decorators import auto_move_data
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, CosineAnnealingLR, MultiStepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics.classification import BinaryF1Score
from torchmetrics.functional import accuracy
from warnings import warn
from copy import copy
import importlib

from lightning.data.dataset_utils import batch_unpack_function
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from torch_metric_learning.noise_reducers import noise_reducers
from lightning.models.utils import *


def class_from_string(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class DecoupledLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self._train_dataloader is None:
            self._train_dataloader = self.trainer.datamodule.train_dataloader()
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self._val_dataloader is None:
            self._val_dataloader = self.trainer.datamodule.val_dataloader()
        return self._val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self._test_dataloader is None:
            self._test_dataloader = self.trainer.datamodule.test_dataloader()
        return self._test_dataloader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()


# TODO: https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch
class LitModelBase(DecoupledLightningModule):
    def __init__(self,
                 batch_size=256,
                 num_classes=1000,
                 num_channels=3,
                 use_pretrained_weights=True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[Type[_LRScheduler]] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 loss_class: Union[
                     Type[losses.BaseMetricLossFunction], Type[losses.BaseLossWrapper]] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer_class: Type[noise_reducers.DefaultNoiseReducer] = noise_reducers.DefaultNoiseReducer,
                 noise_reducer_kwargs: Optional[Dict] = None,
                 loss_weights=None
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone, self.classification_head, self.embedding_size = \
            self._create_model()

        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        refined_loss_kwargs, cross_batch_memory_kwargs = self._populate_loss_args(loss_class, loss_kwargs)
        # refined_loss_kwargs['reducer'] = construct_reducer(reducer_class=reducers.AvgNonZeroReducer)

        self.miner = construct_miner(miner_class, self._populate_miner_args(miner_class, miner_kwargs), distance=None)

        if cross_batch_memory_kwargs is not None:
            metric_loss = loss_class(**refined_loss_kwargs)
            self.metric_loss = losses.CrossBatchMemory(metric_loss,
                                                       embedding_size=self.embedding_size,
                                                       memory_size=cross_batch_memory_kwargs['memory_size'],
                                                       miner=self.miner)
        else:
            self.metric_loss = loss_class(**refined_loss_kwargs)

        # TODO: Remove after inspection that is not needed / weighted was done within reducer instead
        # if type(self.miner).__name__ == 'PopulationAwareMiner' and \
        #         self.miner.pair_rejection_strategy.use_raw_probabilities:
        #     try:
        #         patch_object_with_distance(self.metric_loss)
        #     except Exception:
        #         warn('Unsupported distance patching !!!!!!!!!!')

        self.noise_reducer = \
            construct_noise_reducer(noise_reducer_class, noise_reducer_kwargs,
                                    embedding_size=self.embedding_size,
                                    num_classes=self.hparams.num_classes,
                                    cross_batch_memory_object=self.metric_loss
                                    if cross_batch_memory_kwargs is not None else None)

        # Possibly initialized in fit start. Check documentation there.
        self._alternative_train_dataloader = None

        self.loss_weights = loss_weights if loss_weights is not None else {'classification': 0.0, 'metric': 1.0}

        self.batch_unpack_fn = batch_unpack_function

    def _create_model(self) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
        raise NotImplementedError

    def _populate_miner_args(self, miner_class, miner_kwargs):
        if miner_kwargs is not None:
            actual_miner_kwargs = copy(miner_kwargs)
        else:
            actual_miner_kwargs = {}

        init_args = get_init_args(miner_class)

        # No longer needed, however I leave it here for future reference
        if 'num_classes' in init_args:
            actual_miner_kwargs['num_classes'] = self.hparams.num_classes

        return actual_miner_kwargs

    def _populate_loss_args(self, loss_class, loss_kwargs) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Populates a dictionary which contains loss arguments. Some of them are given, some other are computed or bound
        to the data

        :param loss_class: Loss class
        :param loss_kwargs: Given keyword arguments
        :param num_classes: Number of classes
        :return:
        """
        # With specifying distance = None we let the default distance.
        # Some losses work only with specific distances and letting the user define it may be error-prone
        distance = None
        # Same with reducer
        reducer = None  # reducers.ThresholdReducer(low=0)

        # We get a copy of the loss args, because some of them are not for the loss specifically but for the optimizer
        # of the loss, e.g. in SoftTripleLoss. But also some
        if loss_kwargs is not None:
            actual_loss_kwargs = copy(loss_kwargs)
        else:
            actual_loss_kwargs = {}

        try:
            cross_batch_memory_kwargs = actual_loss_kwargs.pop('cross_batch_memory')
        except KeyError:
            cross_batch_memory_kwargs = None

        # Arguments required for some losses e.g., SoftTriple loss
        # Inspect the signature of the loss and parent classes in order to realize if these arguments are needed.
        # This is to avoid unexpected keyword argument errors
        init_args = get_init_args(loss_class)

        for key in list(actual_loss_kwargs.keys()):
            if key not in init_args:
                del actual_loss_kwargs[key]

        # in some losses these parameters are obtained from kwargs e.g. in
        # https://github.com/KevinMusgrave/pytorch-metric-learning/blob/10fd517a29d86d7752d0bdd7e0864377b89b3fc3
        # /src/pytorch_metric_learning/losses/subcenter_arcface_loss.py#L15
        if 'num_classes' in init_args or loss_class == losses.SubCenterArcFaceLoss:
            actual_loss_kwargs['num_classes'] = self.hparams.num_classes
        if 'embedding_size' in init_args or loss_class == losses.SubCenterArcFaceLoss:
            actual_loss_kwargs['embedding_size'] = self.embedding_size
        if 'distance' in init_args:
            actual_loss_kwargs['distance'] = distance
        if 'reducer' in init_args:
            actual_loss_kwargs['reducer'] = reducer

        if 'loss' in actual_loss_kwargs:
            # TODO: For now I leave it with no / default args
            actual_loss_kwargs['loss'] = class_from_string(actual_loss_kwargs['loss'])()

        return actual_loss_kwargs, cross_batch_memory_kwargs

    def forward(self, x, return_feat=False):
        feats = self.backbone(x)
        logits = self.classification_head(feats)

        # Any normalization should be applied in loss level, e.g. as in losses defined by pytorch-metric-learning
        if return_feat:
            return logits, feats
        else:
            return logits

    def forward_features(self, x):
        feats = self.backbone(x)
        return feats

    def _compute_metric_loss(self, feat, y, mined_indices):
        # Required step in case the miner has a distance weighting according to the mining logic
        if self.noise_reducer is not None and self.noise_reducer.strategy.use_raw_probabilities:
            # TODO: FIX THIS AWKWARD LOGIC
            try:
                # self.metric_loss.distance.weights = self.miner.pair_rejection_strategy.retrieve_batch_weights()
                self.metric_loss.reducer.weights = self.noise_reducer.strategy.retrieve_batch_weights().to('cuda:0')
                self.metric_loss.reducer.mined_indices = mined_indices
            except Exception:
                print('This metric loss does not support reducing')

        # Check if the loss is wrapped
        if isinstance(self.metric_loss, CrossBatchMemory) and self.noise_reducer is not None and \
                not self.noise_reducer.memory_is_dynamic():
            # The noise reducer has already enqueued the batch stuff in cross batch memory,
            # therefore we entirely omit enqueueing through the loss forward method
            return self.metric_loss(feat, y, enqueue_mask=torch.zeros_like(y, dtype=torch.bool))

        return self.metric_loss(feat, y, mined_indices)

    def _compute_classification_loss(self, logits, y):
        return self.classification_loss(logits, y)

    def _compute_loss(self, feat, logits, y, mined_tuples):
        metric_loss = self._compute_metric_loss(feat, y, mined_tuples) if self.loss_weights['metric'] != 0 else 0.0
        classification_loss = self._compute_classification_loss(logits, y) if self.loss_weights[
                                                                                  'classification'] != 0 else 0.0

        self.log("train_loss_metric", metric_loss)
        self.log("train_loss_classification", classification_loss)

        total_loss = (self.loss_weights['classification'] * classification_loss +
                      self.loss_weights['metric'] * metric_loss)

        self.log("train_loss_total", total_loss)

        return total_loss

    def _remove_noisy_from_batch(self, batch, feat, y, logits):
        # Try to get the ground truth noisy information. This is used by the Dummy noise reducer for its reduction
        # and from other reducer for debugging / comparing against gt
        try:
            gt_noisy = self.batch_unpack_fn(batch, keys=('is_noisy',))[0]
        except (KeyError, TypeError):
            gt_noisy = None

        try:
            dataset_indices = self.batch_unpack_fn(batch, keys=('data_idx',))[0]
        except (KeyError, TypeError):
            dataset_indices = None

        batch_noisy_predictions = self.noise_reducer(feat, y, noisy_samples=gt_noisy, dataset_indices=dataset_indices,
                                                     logits=logits)

        with_relabel = isinstance(batch_noisy_predictions, tuple)

        if with_relabel:
            warn(f'Relabeling {self.trainer.current_epoch}')

            batch_noisy_predictions, updated_labels, keep_mask = batch_noisy_predictions
            # god_labels = y.clone()
            # god_labels[batch_noisy_predictions] = batch['target_orig'][batch_noisy_predictions]

            clean_feat = feat[keep_mask]
            clean_y = updated_labels[keep_mask]
            clean_logits = logits[keep_mask]

            self.log("percent_kept_from_relabeling", (torch.count_nonzero(keep_mask) / len(keep_mask)).item())
        else:
            batch_clean_predictions = torch.logical_not(batch_noisy_predictions)
            clean_feat = feat[batch_clean_predictions]
            clean_y = y[batch_clean_predictions]
            clean_logits = logits[batch_clean_predictions]

        batch_clean_predictions = torch.logical_not(batch_noisy_predictions)
        f1_score_metric = BinaryF1Score().cuda()

        # TODO: The below does not work for SCN due to relabeling and wrong F1 is reported.
        #  Need to change this to get the batch noisy targets directly
        #f1_score = f1_score_metric(batch_clean_predictions.cuda(), batch['target_orig'] == batch['target'])
        # The fix:
        f1_score = f1_score_metric(batch_clean_predictions.cuda(), torch.logical_not(batch['is_noisy']))
        self.log("noisy_pred_f1", f1_score)

        all_probs_max = torch.nn.functional.softmax(logits, dim=1).max(dim=1)
        max_values, max_indices = all_probs_max
        self.log("cls_prob_avg_max", max_values.mean())

        return clean_feat, clean_y, clean_logits

    def training_step(self, batch, batch_idx):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'target'))

        logits, feat = self(x, return_feat=True)

        if self.noise_reducer is not None:
            clean_feat, clean_y, clean_logits = self._remove_noisy_from_batch(batch, feat, y, logits)

            # Ignore the cleanup
            # TODO: implement logic for more tolerant noise reduction as a second step
            if len(clean_feat) != 0:
                feat = clean_feat
                y = clean_y
                logits = clean_logits

        if len(feat) != 0:
            indices_tuple = self.miner(feat, y)
            assert indices_tuple is not None
        else:
            warn('Processing zero dimensional batch after noise reduction')
            indices_tuple = lmu.get_all_pairs_indices(y)

        total_loss = self._compute_loss(feat, logits, y, indices_tuple)
        return total_loss

    def _eval_step(self, batch, stage=None):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'target'))
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self._compute_classification_loss(logits, y)
        acc = accuracy(preds, y, task='multiclass', num_classes=self.hparams.num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def on_fit_start(self):
        # If we have a logger and a noise reducer, patch the output path of noise reducer from logger's path
        if (self.logger is not None and
                self.noise_reducer is not None and self.noise_reducer.inspector is not None):
            logging_dir = self.logger.log_dir
            output_path = Path(logging_dir) / 'rejection_inspector_output'
            output_path.mkdir(exist_ok=True)
            self.noise_reducer.inspector._output_dir = output_path

        # Here we need to define a train dataloader which is the same as the original but without any sampler or maybe
        # even randomization on the order. That is because we need to extract features from all samples of the dataset,
        # and in many occasions random samplers such as RandomIdentitySampler omit samples of the dataset in order to
        # formulate batches that meet certain criteria.
        if self.noise_reducer is not None and self.noise_reducer.with_epoch_features:
            original_train_dataloader = self.train_dataloader()
            self._alternative_train_dataloader = DataLoader(dataset=original_train_dataloader.dataset,
                                                            batch_size=original_train_dataloader.batch_size,
                                                            shuffle=False,
                                                            num_workers=original_train_dataloader.num_workers,
                                                            drop_last=False,
                                                            pin_memory=original_train_dataloader.pin_memory,
                                                            sampler=None)

    def on_train_start(self):
        # In this case we want the features from the pretrained model. We check the following cases:
        # (a) miner is PopulationAwareMiner, which is the only one supporting the option
        # (b) 'use_pretrained' is enabled
        # (c) model is loaded with pretrained weights
        if self.noise_reducer is not None and self.noise_reducer.use_pretrained and \
                self.hparams.use_pretrained_weights is True:
            print('Extracting features from pretrained model...')
            all_features, all_class_labels, all_data_indices = self._extract_features(self.train_dataloader())
            self.noise_reducer.bootstrap_initial(all_features, all_class_labels, dataset_indices=all_data_indices)

    def training_epoch_end(self, training_step_outputs):
        if self.noise_reducer is not None:
            if not self.noise_reducer.with_epoch_features or not self.noise_reducer.use_current_epoch_features():
                self.noise_reducer.bootstrap_epoch(self.trainer.current_epoch)
            else:
                print('Extracting features from model at current epoch...')
                all_features, all_class_labels, all_data_indices = (
                    self._extract_features(self._alternative_train_dataloader))

                self.noise_reducer.bootstrap_epoch(self.trainer.current_epoch, all_features, all_class_labels,
                                                   all_data_indices)

        # lr_info_dict = {}
        # for pg in self.trainer.optimizers[0].param_groups:
        #     lr_info_dict[pg['name']] = pg['lr']
        #
        # print(lr_info_dict)

    @property
    def num_training_steps(self) -> int:
        """
        Implementation from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449

        @return: Total training steps inferred from datamodule and devices.
        """
        warn(
            'The implementation may lack edge cases. See https://github.com/PyTorchLightning/pytorch-lightning/issues/5449')

        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        train_dataloader = self.train_dataloader()

        if hasattr(train_dataloader.sampler, 'n_iterations'):
            if train_dataloader.sampler.n_iterations > 0:
                return train_dataloader.sampler.n_iterations
            else:
                raise MisconfigurationException('Required steps are not provided by the sampler.')

        limit_batches = self.trainer.limit_train_batches
        batches = len(train_dataloader)
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def _extract_features(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        first_batch = next(iter(dataloader))
        images, targets, data_indices = self.batch_unpack_fn(first_batch, keys=('image', 'target', 'data_idx'))
        feats = self.forward_features(images.to(device))

        n_samples = len(dataloader.dataset)
        batch_size = feats.shape[0]

        all_features = torch.zeros((n_samples, feats.shape[1]), dtype=feats.dtype)
        all_class_labels = torch.zeros((n_samples, ), dtype=targets.dtype)
        all_data_indices = torch.zeros((n_samples, ), dtype=data_indices.dtype)

        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                x, y, idx = self.batch_unpack_fn(batch, keys=('image', 'target', 'data_idx'))
                _, feat = self(x.to(device), return_feat=True)

                data_chunk_idx = slice(i * batch_size, (i + 1) * batch_size)

                all_features[data_chunk_idx] = feat.detach().cpu()
                all_class_labels[data_chunk_idx] = y
                all_data_indices[data_chunk_idx] = idx

        return all_features, all_class_labels, all_data_indices

    def _construct_scheduler(self, optimizer, scheduler_class, scheduler_kwargs):
        """
        Some schedulers like OneCycleLR and CosineAnnealingLR require parameters that are not  known beforehand, but
        depend on runtime. E.g. number of steps can be obtained from the dataloader

        :param scheduler_class:
        :return:
        """

        # Regard None kwargs as empty dict
        scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs

        # Construct a scheduler config, rather than returning a single scheduler in order to configure other params
        # such as interval etc
        scheduler_config = \
            {}

        # Special check for few schedulers
        if scheduler_class.__name__ == 'OneCycleLR':
            scheduler_config['interval'] = 'step'
            # Assign 0.1*lr as shown in OneCycleLR example, don't know why it is done like this
            # max_lr_list = [0.1 * pg['lr'] for pg in optimizer.param_groups]
            # Assign the lr itself as done in unicom
            max_lr_list = [pg['lr'] for pg in optimizer.param_groups]
            scheduler_config['scheduler'] = OneCycleLR(optimizer, max_lr_list, total_steps=self.num_training_steps)
        elif scheduler_class.__name__ == 'CosineAnnealingLR':
            scheduler_config['interval'] = 'step'
            scheduler_config['scheduler'] = CosineAnnealingLR(optimizer, T_max=self.num_training_steps)
        elif scheduler_class.__name__ == 'MultiStepLR':
            # For some reason if I set interval to epoch it does not work
            scheduler_config['interval'] = 'step'
            if 'milestones' in scheduler_kwargs:
                steps_per_epoch = self.num_training_steps / self.trainer.max_epochs
                scheduler_kwargs['milestones'] = [m * steps_per_epoch for m in scheduler_kwargs['milestones']]
            scheduler_config['scheduler'] = MultiStepLR(optimizer, **scheduler_kwargs)
        elif scheduler_class.__name__ == 'LinearWarmupCosineAnnealingLR':
            # For some reason if I set interval to epoch it does not work
            scheduler_config['interval'] = 'step'
            steps_per_epoch = self.num_training_steps / self.trainer.max_epochs
            if 'warmup_epochs' in scheduler_kwargs:
                scheduler_kwargs['warmup_epochs'] = steps_per_epoch * scheduler_kwargs['warmup_epochs']
            if 'max_epochs' in scheduler_kwargs:
                scheduler_kwargs['max_epochs'] = steps_per_epoch * scheduler_kwargs['max_epochs']

            scheduler_config['scheduler'] = LinearWarmupCosineAnnealingLR(optimizer, **scheduler_kwargs)
        else:
            scheduler_config['scheduler'] = scheduler_class(optimizer, **scheduler_kwargs)

        return scheduler_config

    def _construct_direct_children_param_groups(self, lr_overrides, model_optimizer_kwargs):
        # Expected format for lr_overrides with type 'direct_children':
        # [
        # {
        #       'type': 'direct_children',
        #       'overrides':
        #       [
        #           {
        #               'children':
        #               [
        #                   'backbone.parts_avgpool',
        #                   'backbone.dropout',
        #                   'backbone.conv5',
        #                   'classification_head'
        #               ],
        #               'lr_multiplier': 10,
        #           }
        #       ]
        # }
        # ]
        #
        # Notice that modules of the backbone need to have a prepended 'backbone.' string

        model_param_groups = []

        # The following dict is in the form <module_name> -> <module>
        # module names are python attributes and therefore their names are unique

        model_children_modules = {}
        for name, module in self.backbone.named_children():
            model_children_modules[f'backbone.{name}'] = module

        model_children_modules['classification_head'] = self.classification_head

        model_children_modules_names = set(model_children_modules.keys())

        # First we obtain the overrides
        for i, lr_override in enumerate(lr_overrides['overrides']):
            grouped_params = []

            # I intentionally leave all possible errors unhandled, e.g. wrong module name, double declaration etc...
            for child_module_name in lr_override['children']:
                grouped_params.extend(list(model_children_modules[child_module_name].parameters()))
                model_children_modules_names.remove(child_module_name)

            model_optimizer_kwargs_copy = copy(model_optimizer_kwargs)
            model_optimizer_kwargs_copy['lr'] = (
                    lr_override['lr_multiplier'] * model_optimizer_kwargs_copy['lr'])
            model_param_groups.append(
                {'params': grouped_params, 'name': f'custom_lr_param_group-{i}', **model_optimizer_kwargs_copy})

        # The rest modules should use the default lr
        grouped_params = []
        for child_module_name in model_children_modules_names:
            grouped_params.extend(list(model_children_modules[child_module_name].parameters()))

        if len(grouped_params) > 0:
            model_param_groups.insert(0, {'params': grouped_params, 'name': 'default_lr_param_group',
                                          **model_optimizer_kwargs})

        return model_param_groups

    def _construct_arbitrary_children_param_groups(self, lr_overrides, model_optimizer_kwargs):
        model_param_groups = []

        matched_keys = set()

        for i, lr_override in enumerate(lr_overrides['overrides']):
            grouped_params = []
            for child_param_name in lr_override['children']:
                for key, value in self.named_parameters():
                    if not value.requires_grad:
                        continue
                    if child_param_name in key:
                        if key in matched_keys:
                            raise AssertionError(f'Param {key} already set!')
                        else:
                            matched_keys.add(key)

                        grouped_params.append(value)

            model_optimizer_kwargs_copy = copy(model_optimizer_kwargs)
            model_optimizer_kwargs_copy['lr'] = (
                    lr_override['lr_multiplier'] * model_optimizer_kwargs_copy['lr'])
            model_param_groups.append(
                {'params': grouped_params, 'name': f'custom_lr_param_group-{i}', **model_optimizer_kwargs_copy})

        # The rest modules should use the default lr
        grouped_params = []
        for key, value in self.named_parameters():
            if key not in matched_keys:
                grouped_params.append(value)

        if len(grouped_params) > 0:
            model_param_groups.insert(0, {'params': grouped_params, 'name': 'default_lr_param_group',
                                          **model_optimizer_kwargs})

        return model_param_groups

    def configure_optimizers(self):
        def ensure_type(class_type):
            if isinstance(class_type, str):
                module, class_name = class_type.rsplit('.', maxsplit=1)
                return getattr(importlib.import_module(module), class_name)
            else:
                return class_type

        optimizer_class = ensure_type(self.hparams.optimizer_class)

        # Construct the optimizer
        model_optimizer_kwargs = self.hparams.optimizer_kwargs

        lr_overrides = model_optimizer_kwargs.pop('lr_overrides', None)
        if lr_overrides is None:

            # For now I leave it as separate groups in case I later prefer to have a default multiplied lr for the
            # classification head. If the metric loss has trainable parameters they have to be defined explicitly in
            # loss_kwargs dictionary
            model_param_groups = \
                [
                    {'params': self.backbone.parameters(), 'name': 'backbone_param_group', **model_optimizer_kwargs},
                    {'params': self.classification_head.parameters(), 'name': 'cls_param_group',
                     **model_optimizer_kwargs}
                ]
        else:
            if lr_overrides['type'] == 'direct_children':
                model_param_groups = self._construct_direct_children_param_groups(lr_overrides, model_optimizer_kwargs)
            elif lr_overrides['type'] == 'arbitrary_children':
                model_param_groups = self._construct_arbitrary_children_param_groups(lr_overrides,
                                                                                     model_optimizer_kwargs)
            else:
                raise NotImplementedError

        # Look for existence of optimizer_kwargs in loss_args
        if self.hparams.loss_kwargs is None or 'optimizer_kwargs' not in self.hparams.loss_kwargs:
            param_groups = \
                [
                    *model_param_groups,
                ]
            optimizer = optimizer_class(param_groups, **model_optimizer_kwargs)
        else:
            loss_optimizer_kwargs = self.hparams.loss_kwargs['optimizer_kwargs']
            default_optimizer_kwargs = {}

            # We keep rest keys as default values in the optimizer
            for k, v in copy(model_optimizer_kwargs).items():
                if k not in loss_optimizer_kwargs:
                    default_optimizer_kwargs[k] = v
                    del model_optimizer_kwargs[k]
            param_groups = \
                [
                    *model_param_groups,
                    {'params': self.metric_loss.parameters(), 'name': 'metric_loss_param_group',
                     **loss_optimizer_kwargs}
                ]
            optimizer = optimizer_class(param_groups, **default_optimizer_kwargs)

        optimizer_dict = {'optimizer': optimizer}

        if self.hparams.scheduler_class is not None:
            scheduler_class = ensure_type(self.hparams.scheduler_class)
            scheduler_kwargs = self.hparams.scheduler_kwargs

            optimizer_dict['lr_scheduler'] = self._construct_scheduler(optimizer, scheduler_class, scheduler_kwargs)

        return optimizer_dict
