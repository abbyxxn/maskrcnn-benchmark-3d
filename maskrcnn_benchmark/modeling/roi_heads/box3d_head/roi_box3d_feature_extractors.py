# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

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

        return x


class Box3dPCFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(Box3dPCFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX3D_HEAD.POOLER_RESOLUTION
        scales = (1.,)
        sampling_ratio = cfg.MODEL.ROI_BOX3D_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

    def forward(self, proposal_per_image, pseudo_pc):
        pointcloud = torch.unsqueeze(pseudo_pc, 0)
        pointclouds = (pointcloud,)
        proposal = [proposal_per_image, ]
        x = self.pooler(pointclouds, proposal)
        return x


_ROI_BOX3D_FEATURE_EXTRACTORS = {
    "Box3dFeatureExtractor": Box3dFeatureExtractor,
    "Box3dPCFeatureExtractor": Box3dPCFeatureExtractor,
}


def make_roi_box3d_feature_extractor(cfg):
    func = _ROI_BOX3D_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX3D_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)


def make_roi_pc_feature_extractor(cfg):
    func = _ROI_BOX3D_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX3D_HEAD.POINT_CLOUD_FEATURE_EXTRACTOR]
    return func(cfg)
