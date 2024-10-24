"""
Some proxy classes imported and modified from https://github.com/tinyvision/SOLIDER-REID/blob/master/model/make_model.py
"""
import torch
import torch.nn as nn
import copy
from yacs.config import CfgNode as CN
from lightning.models.backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, \
    swin_tiny_patch4_window7_224
import types


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        convert_weights = True if pretrain_choice == 'imagenet' else False
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        # Converted simple path to cfg as needed by transformer, else
                                                        # it throws deprecation warning
                                                        init_cfg=dict(type='Pretrained', checkpoint=model_path),
                                                        convert_weights=convert_weights,
                                                        semantic_weight=semantic_weight)
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat, featmaps = self.base(x)

        # For now I omit the featmaps. Not used in solider
        return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


__factory_T_type = {
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
}


def swin_transformer_default_config():
    _C = CN()
    # -----------------------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------------------
    _C.MODEL = CN()
    # Using cuda or cpu for training
    _C.MODEL.DEVICE = "cuda"
    # ID number of GPU
    _C.MODEL.DEVICE_ID = '0'
    # Name of backbone
    _C.MODEL.NAME = 'transformer'
    # Last stride of backbone
    _C.MODEL.LAST_STRIDE = 1
    # Path to pretrained model of backbone
    _C.MODEL.PRETRAIN_PATH = '/path/to/pretrained/weights'
    _C.MODEL.PRETRAIN_HW_RATIO = 2

    # Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
    # Options: 'imagenet' , 'self' , 'finetune'
    _C.MODEL.PRETRAIN_CHOICE = 'self'

    # If train with BNNeck, options: 'bnneck' or 'no'
    _C.MODEL.NECK = 'bnneck'
    # If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
    _C.MODEL.IF_WITH_CENTER = 'no'

    _C.MODEL.ID_LOSS_TYPE = 'softmax'
    _C.MODEL.ID_LOSS_WEIGHT = 1.0
    _C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

    _C.MODEL.METRIC_LOSS_TYPE = 'triplet'
    # If train with multi-gpu ddp mode, options: 'True', 'False'
    _C.MODEL.DIST_TRAIN = False
    # If train with soft triplet loss, options: 'True', 'False'
    _C.MODEL.NO_MARGIN = True
    # If train with label smooth, options: 'on', 'off'
    _C.MODEL.IF_LABELSMOOTH = 'off'
    _C.MODEL.IF_LABELSMOOTH = 'off'
    # If train with arcface loss, options: 'True', 'False'
    _C.MODEL.COS_LAYER = False

    _C.MODEL.DROPOUT_RATE = 0.0
    # Reduce feature dim
    _C.MODEL.REDUCE_FEAT_DIM = False
    _C.MODEL.FEAT_DIM = 512
    # Transformer setting
    _C.MODEL.DROP_PATH = 0.1
    _C.MODEL.DROP_OUT = 0.0
    _C.MODEL.ATT_DROP_RATE = 0.0
    _C.MODEL.TRANSFORMER_TYPE = 'swin_tiny_patch4_window7_224'
    _C.MODEL.STRIDE_SIZE = [16, 16]
    _C.MODEL.GEM_POOLING = False
    _C.MODEL.STEM_CONV = False

    # JPM Parameter
    _C.MODEL.JPM = False
    _C.MODEL.SHIFT_NUM = 5
    _C.MODEL.SHUFFLE_GROUP = 2
    _C.MODEL.DEVIDE_LENGTH = 4
    _C.MODEL.RE_ARRANGE = True

    # SIE Parameter
    _C.MODEL.SIE_COE = 3.0
    _C.MODEL.SIE_CAMERA = False
    _C.MODEL.SIE_VIEW = False

    # Semantic Weight
    _C.MODEL.SEMANTIC_WEIGHT = 0.2

    _C.TEST = CN()
    _C.TEST.NECK_FEAT = 'before'

    _C.INPUT = CN()
    _C.INPUT.SIZE_TRAIN = [384, 128]

    return _C


def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
    if cfg.MODEL.JPM:
        raise NotImplementedError
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
    print('===========building transformer===========')
    return model


def main():
    cfg = swin_transformer_default_config()

    cfg.MODEL.PRETRAIN_PATH = '/media/amidemo/Data/object_classifier_data/model_zoo/solider_models/swin_tiny_tea.pth'

    model = make_model(cfg, num_class=751, camera_num=None, view_num=None, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.cuda()

    batch_size = 32
    random_input = torch.rand((batch_size, 3, 384, 128))
    random_labels = torch.randint(0, 751, (batch_size,))
    outputs = model(random_input.cuda(), random_labels.cuda())
    pass


if __name__ == '__main__':
    main()
