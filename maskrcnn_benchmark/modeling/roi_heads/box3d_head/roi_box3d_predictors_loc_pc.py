# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn


class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class Box3dLocPCPredictor(nn.Module):
    def __init__(self, cfg):
        super(Box3dLocPCPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        input_size = cfg.MODEL.ROI_BOX3D_HEAD.POINTCLOUD_OUT_CHANNELS * (
                cfg.MODEL.ROI_BOX3D_HEAD.POOLER_RESOLUTION ** 2)
        representation_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_HEAD_DIM

        self.fc6 = nn.Linear(input_size, representation_size)
        self.bbox3d_dimension_pred = nn.Linear(representation_size, num_classes * 3)

        for l in [self.fc6, ]:
            nn.init.kaiming_uniform_(self.bbox3d_dimension_pred.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        bbox3d_dimension_deltas = self.bbox3d_dimension_pred(x)

        return bbox3d_dimension_deltas


_ROI_BOX3D_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "Box3dLocPCPredictor": Box3dLocPCPredictor,
}


def make_roi_box3d_predictor_localization_pc(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR_LOCALIZATION_PC]
    return func(cfg)
