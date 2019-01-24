# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn

from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler


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
    "Box3dFeatureExtractor": Box3dFeatureExtractor,
}


def make_roi_box3d_feature_extractor(cfg):
    func = _ROI_BOX3D_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX3D_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
