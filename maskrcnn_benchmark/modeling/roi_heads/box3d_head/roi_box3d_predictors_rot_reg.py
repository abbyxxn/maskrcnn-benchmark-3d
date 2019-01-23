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


class RotationRegressionPredictor(nn.Module):
    def __init__(self, cfg):
        super(RotationRegressionPredictor, self).__init__()
        self.num_bins = cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_BIN
        input_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_HEAD_DIM

        self.bbox3d_rotation_regression_pred = nn.Linear(input_size, self.num_bins * 2)

        nn.init.normal_(self.bbox3d_rotation_regression_pred.weight, std=0.001)
        for l in [self.bbox3d_rotation_regression_pred, ]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        bbox3d_rotation_regression_deltas = self.bbox3d_rotation_regression_pred(x)

        return bbox3d_rotation_regression_deltas


_ROI_BOX3D_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "RotationRegressionPredictor": RotationRegressionPredictor,
}


def make_roi_box3d_predictor_rotation_regression(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR_ROTATION_REGRESSION]
    return func(cfg)
