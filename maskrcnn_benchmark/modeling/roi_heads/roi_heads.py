# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box3d_head.box3d_head import build_roi_box3d_head
from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.BOX3D_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.box3d.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, img_original_ids=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        roi_box_features, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                    self.training
                    and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = roi_box_features
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        if self.cfg.MODEL.BOX3D_ON:
            box3d_features = features
            x, detections, loss_box3d = self.box3d(box3d_features, detections, targets,
                                                   img_original_ids=img_original_ids)
            losses.update(loss_box3d)
        return roi_box_features, detections, losses


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))
    if cfg.MODEL.BOX3D_ON:
        roi_heads.append(("box3d", build_roi_box3d_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
