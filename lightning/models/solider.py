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
from lightning.models.backbones.solider_reid_proxies import make_model, swin_transformer_default_config
from lightning.models.utils import weights_init_kaiming, weights_init_classifier, weights_init_xavier
from yacs.config import CfgNode as CN


class LitSolider(LitModelBase):
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
                 loss_class: Union[
                     Type[losses.BaseMetricLossFunction], Type[losses.BaseLossWrapper]] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer_class: Type[noise_reducers.DefaultNoiseReducer] = noise_reducers.DefaultNoiseReducer,
                 noise_reducer_kwargs: Optional[Dict] = None,
                 loss_weights=None,
                 # Custom options
                 model_variant: Literal[
                     'swin_base_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_tiny_patch4_window7_224'] =
                 'swin_tiny_patch4_window7_224',
                 pretrained_weights_path='',
                 cls_head_dropout_rate=0.0,
                 reduced_feat_dim=None,
                 test_feat_after_bn=False
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

        # Explicit override
        self.loss_weights = {'classification': 1.0, 'metric': 1.0}

    def _create_model(self) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
        """

        @return: backbone and linear layer
        """

        cfg = swin_transformer_default_config()

        # Early check that the path is not None/empty because it causes the later code to simply not load the weights
        # rather than throwing an error
        assert self.hparams.pretrained_weights_path is not None and self.hparams.pretrained_weights_path != ''

        cfg.MODEL.PRETRAIN_PATH = self.hparams.pretrained_weights_path

        try:
            # TODO: The below creates a thin wrapper over swin transformer using the default parameters from solider
            #  implementation. Move all the code and options here and directly construct the base model instead of the
            #  wrapper.
            model = make_model(cfg,
                               num_class=self.hparams.num_classes,
                               camera_num=None,
                               view_num=None,
                               semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)

        except (KeyError, Exception) as e:
            raise NotImplementedError('Unsupported SOLIDER variant')

        if self.hparams.reduced_feat_dim is not None:
            fcneck = nn.Linear(model.in_planes, self.hparams.reduced_feat_dim, bias=False)
            fcneck.apply(weights_init_xavier)

            model = nn.Sequential(model, fcneck)

            embedding_size = self.hparams.reduced_feat_dim
        else:
            embedding_size = model.in_planes

        # Construct the classification layer
        bottleneck = nn.BatchNorm1d(embedding_size)
        bottleneck.bias.requires_grad_(False)
        bottleneck.apply(weights_init_kaiming)

        dropout = nn.Dropout(self.hparams.cls_head_dropout_rate)

        classifier = nn.Linear(model.in_planes, self.hparams.num_classes, bias=False)
        classifier.apply(weights_init_classifier)

        # TODO: SOLIDER has support for additional classifiers other than linear, arcface, cosface, etc however this
        #  linear is actually used so I didn't bother supporting the rest. Possibly leftovers from transreid.
        fc = nn.Sequential(
            bottleneck,
            dropout,
            classifier
        )

        return model, fc, embedding_size

    def forward(self, x, return_feat=False):
        feats = self.backbone(x)

        if self.training:
            logits = self.classification_head(feats)

            # Any normalization should be applied in loss level, e.g. as in losses defined by pytorch-metric-learning
            if return_feat:
                return logits, feats
            else:
                return logits
        else:
            if self.test_feat_after_bn:
                # The BN is the first module - index 0
                return self.classification_head[0](feats)
            else:
                return feats


def check_model():
    pretrain_path = '/media/amidemo/Data/object_classifier_data/model_zoo/solider_models/swin_tiny_tea.pth'
    solider_model = LitSolider(pretrained_weights_path=pretrain_path)

    solider_model.cuda()

    batch_size = 32
    random_input = torch.rand((batch_size, 3, 384, 128))
    random_labels = torch.randint(0, 751, (batch_size,))
    outputs = solider_model(random_input.cuda(), return_feat=True)

    pass


if __name__ == '__main__':
    check_model()
