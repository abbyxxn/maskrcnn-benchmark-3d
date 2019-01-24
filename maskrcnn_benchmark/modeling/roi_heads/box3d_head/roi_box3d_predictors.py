# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F


class Box3DPredictor(nn.Module):
    def __init__(self, cfg):
        super(Box3DPredictor, self).__init__()
        input_size = (cfg.MODEL.BACKBONE.OUT_CHANNELS + cfg.MODEL.ROI_BOX3D_HEAD.POINTCLOUD_OUT_CHANNELS) * (
                cfg.MODEL.ROI_BOX3D_HEAD.POOLER_RESOLUTION ** 2)
        representation_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_HEAD_DIM

        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        for l in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class Box3dDimPredictor(nn.Module):
    def __init__(self, cfg):
        super(Box3dDimPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        input_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_HEAD_DIM

        self.bbox3d_dimension_pred = nn.Linear(input_size, num_classes * 3)

        nn.init.normal_(self.bbox3d_dimension_pred.weight, std=0.001)
        for l in [self.bbox3d_dimension_pred, ]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        bbox3d_dimension_deltas = self.bbox3d_dimension_pred(x)

        return bbox3d_dimension_deltas


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


class Box3dLocConvPredictor(nn.Module):
    def __init__(self, cfg):
        super(Box3dLocConvPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        input_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_HEAD_DIM

        self.bbox3d_dimension_pred = nn.Linear(input_size, num_classes * 3)

        nn.init.normal_(self.bbox3d_dimension_pred.weight, std=0.001)
        for l in [self.bbox3d_dimension_pred, ]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        bbox3d_dimension_deltas = self.bbox3d_dimension_pred(x)

        return bbox3d_dimension_deltas


class RotationPredictor(nn.Module):
    def __init__(self, cfg):
        super(RotationPredictor, self).__init__()
        num_bins = cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_BIN
        input_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_HEAD_DIM

        self.bbox3d_rotation_conf_score = nn.Linear(input_size, num_bins)
        self.bbox3d_rotation_reg_pred = nn.Linear(input_size, self.num_bins * 2)

        nn.init.normal_(self.bbox3d_rotation_conf_score.weight, std=0.001)
        nn.init.normal_(self.bbox3d_rotation_reg_pred.weight, std=0.001)
        for l in [self.bbox3d_rotation_conf_score, self.bbox3d_rotation_reg_pred, ]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.bbox3d_rotation_conf_score(x)
        rotation_deltas = self.bbox3d_rotation_reg_pred(x)

        return scores, rotation_deltas


_ROI_BOX3D_PREDICTOR = {
    "Box3DPredictor": Box3DPredictor,
    "Box3dDimPredictor": Box3dDimPredictor,
    "Box3dLocPCPredictor": Box3dLocPCPredictor,
    "Box3dLocConvPredictor": Box3dLocConvPredictor,
    "RotationPredictor": RotationPredictor,
}


def make_roi_box3d_predictor(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR]
    return func(cfg)


def make_roi_box3d_predictor_dimension(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR_DIMENSION]
    return func(cfg)


def make_roi_box3d_predictor_localization_pc(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR_LOCALIZATION_PC]
    return func(cfg)


def make_roi_box3d_predictor_localization_conv(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR_LOCALIZATION_CONV]
    return func(cfg)


def make_roi_box3d_predictor_rotation(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR_ROTATION]
    return func(cfg)
