# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn

from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler


class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


# TODO change ResNet50Conv5ROIFeatureExtractor for box3d_feature_extractors

class Box3dFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(Box3dFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX3D_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX3D_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX3D_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        # x = x.view(x.size(0), -1)

        return x


_ROI_BOX3D_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "Box3dFeatureExtractor": Box3dFeatureExtractor,
}


def make_roi_box3d_feature_extractor(cfg):
    func = _ROI_BOX3D_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX3D_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
