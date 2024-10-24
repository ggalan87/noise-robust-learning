from typing import Optional, Type, Dict, Literal, Tuple, Union
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_metric_learning import losses, miners
from lightning.models.model_base import LitModelBase, Flatten
from lightning.models.backbones.blocks import DimReduceLayer
from torch_metric_learning.noise_reducers import noise_reducers
from .backbones.pcb import pcb_p4, pcb_p6


class LitPCB(LitModelBase):
    def __init__(self,
                 # Common model options
                 batch_size=256,
                 num_classes=1000,
                 num_channels=3,
                 use_pretrained_weights=True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[Type[_LRScheduler]] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 loss_class: Union[Type[losses.BaseMetricLossFunction], Type[losses.BaseLossWrapper]] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer_class: Type[noise_reducers.DefaultNoiseReducer] = noise_reducers.DefaultNoiseReducer,
                 noise_reducer_kwargs: Optional[Dict] = None,
                 loss_weights=None,
                 # Custom options
                 model_variant: Literal['pcb_p4', 'pcb_p6'] = 'pcb_p6',
                 ):
        super().__init__(batch_size=batch_size,
                         num_classes=num_classes,
                         num_channels=num_channels,
                         use_pretrained_weights=use_pretrained_weights,
                         loss_class=loss_class,
                         loss_kwargs=loss_kwargs,
                         miner_class=miner_class,
                         miner_kwargs=miner_kwargs,
                         noise_reducer_class=noise_reducer_class,
                         noise_reducer_kwargs=noise_reducer_kwargs,
                         loss_weights=loss_weights)
        self.save_hyperparameters()

        self.loss_weights = {'classification': 1.0, 'metric': 1.0}

    def _create_model(self) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
        """

        @return: backbone and linear layer
        """

        # PCB forward returns either logits if loss = 'softmax' or (logits, features) if loss = 'triplet'
        # I leave the naming scheme as such for now and select triplet in order to get features from the forward pass
        if self.hparams.model_variant == 'pcb_p6':
            model = pcb_p6(num_classes=self.hparams.num_classes, loss='triplet')
        elif self.hparams.model_variant == 'pcb_p4':
            model = pcb_p4(num_classes=self.hparams.num_classes, loss='triplet')
        else:
            raise NotImplementedError('Unsupported PCB variant')

        # TODO: This is the original from PCB, essentially multiple classifiers, meed to implement such logic inside
        #  model_base or create and/or override some method here
        fc = nn.ModuleList(
            [
                nn.Linear(model.feature_dim, self.hparams.num_classes)
                for _ in range(model.parts)
            ]
        )

        # By default this is large
        embedding_size = model.feature_dim * model.parts

        return model, fc, embedding_size

    def forward(self, x, return_feat=False):
        feats = self.backbone(x)

        # Classifier in PCB should operate on separate body parts
        logits = []
        for i in range(self.backbone.parts):
            v_h_i = feats[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.classification_head[i](v_h_i)
            logits.append(y_i)

        # PCB returns multiple featuremaps, one per part and we simply reshape the tensors such that they
        # are concatenated
        aggregated_feats = feats.view(feats.size(0), -1)

        # Any normalization should be applied in loss level, e.g. as in losses defined by pytorch-metric-learning
        if return_feat:
            return logits, aggregated_feats
        else:
            return logits

    def _compute_classification_loss(self, logits, y):
        loss = 0.
        for x in logits:
            loss += self.classification_loss(x, y)
        loss /= len(logits)
        return loss
