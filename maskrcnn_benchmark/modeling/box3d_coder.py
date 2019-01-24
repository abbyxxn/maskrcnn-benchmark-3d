import _pickle as cPickle
import math
import os
import numpy as np

import torch
from maskrcnn_benchmark.data.datasets.kitti.convert_disparity_to_depth import Calib


class Box3dCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, root, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
        self.use_depth_encode = False
        self.root = root

    def encode(self, bbox2d, reference_boxes, labels, img_name, use_2d_project_center=True):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        device = reference_boxes.device

        cache_file = os.path.join(self.root, 'car_typical_dimension_gt.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                typical_dimension = cPickle.load(file)

        ex_lengths = torch.zeros(reference_boxes.shape[0], dtype=torch.float32, device=device)
        ex_heights = torch.zeros(reference_boxes.shape[0], dtype=torch.float32, device=device)
        ex_widths = torch.zeros(reference_boxes.shape[0], dtype=torch.float32, device=device)

        for i, label in enumerate(labels):
            ex_lengths[i], ex_heights[i], ex_widths[i] = typical_dimension[label.item()]

        gt_ry = reference_boxes[:, 0]
        gt_lengths = reference_boxes[:, 1]
        gt_heights = reference_boxes[:, 2]
        gt_widths = reference_boxes[:, 3]
        gt_ctr_x = reference_boxes[:, 4]
        gt_ctr_y = reference_boxes[:, 5]
        gt_ctr_z = reference_boxes[:, 6]

        wl, wh, ww, wx, wy, wz = self.weights

        targets_ry = gt_ry

        if use_2d_project_center:
            calib_dir = os.path.join(self.root, 'training/calib')
            calib = Calib(os.path.join(calib_dir, img_name + '.txt'))
            xyz = torch.stack((gt_ctr_x, gt_ctr_y, gt_ctr_z))
            uv = self.center_project_to_image(xyz, calib.lcam.P)
            targets_dx = torch.as_tensor(uv[0], dtype=torch.float32, device=device) - bbox2d[:, 0].reshape(-1)
            targets_dy = torch.as_tensor(uv[1], dtype=torch.float32, device=device) - bbox2d[:, 1].reshape(-1)
        else:
            targets_dx = gt_ctr_x
            targets_dy = gt_ctr_y

        targets_dz = gt_ctr_z

        targets_dl = wl * torch.log(gt_lengths / ex_lengths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dw = ww * torch.log(gt_widths / ex_widths)

        targets = torch.stack((targets_ry, targets_dl, targets_dh, targets_dw, targets_dx, targets_dy, targets_dz),
                              dim=1)
        return targets

    def decode(self, reference_boxes_3d, labels):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes prediction
            boxes (Tensor): reference boxes. proposal
        """
        device = reference_boxes_3d.device
        cache_file = os.path.join(self.root, 'car_typical_dimension_gt.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                typical_dimension = cPickle.load(file)

        ex_lengths = torch.zeros(reference_boxes_3d.shape[0], dtype=torch.float32, device=device)
        ex_heights = torch.zeros(reference_boxes_3d.shape[0], dtype=torch.float32, device=device)
        ex_widths = torch.zeros(reference_boxes_3d.shape[0], dtype=torch.float32, device=device)

        for i, label in enumerate(labels):
            if label not in [1]:
                continue
            ex_lengths[i], ex_heights[i], ex_widths[i] = typical_dimension[label.item()]

        wl, wh, ww, wx, wy, wz = self.weights
        dl = reference_boxes_3d[:, 0] / wl
        dh = reference_boxes_3d[:, 1] / wh
        dw = reference_boxes_3d[:, 2] / ww

        # Prevent sending too large values into torch.exp()
        dl = torch.clamp(dl, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)

        pred_l = torch.exp(dl) * ex_lengths
        pred_h = torch.exp(dh) * ex_heights
        pred_w = torch.exp(dw) * ex_widths

        pred_boxes = torch.stack((pred_h, pred_w, pred_l), dim=1)

        return pred_boxes

    def center_project_to_image(self, xyz, P):
        xyz = xyz.cpu()
        P = np.asmatrix(P)
        uv = P * np.vstack((xyz, np.ones((1, xyz.shape[1]), dtype=np.float32)))
        uv[:2] = uv[:2] / uv[2]
        return np.asarray(uv[:2])

    def center_decode(self, xyz, img_original_ids, boxes):
        xx = []
        yy = []
        zz = []
        for uv, img_name, box in zip(xyz, img_original_ids, boxes):
            box.convert("xywh")
            calib_dir = os.path.join(self.root, 'training/calib')
            calib = Calib(os.path.join(calib_dir, img_name + '.txt'))
            x = (uv[:, 0] + box.bbox[:, 0]) * uv[:, 2] / calib.lcam.f
            y = (uv[:, 1] + box.bbox[:, 1]) * uv[:, 2] / calib.lcam.f
            xx.append(x)
            yy.append(y)
            zz.append(uv[:, 2])
            box.convert("xyxy")
        xx = torch.cat(xx).reshape(-1, 1)
        yy = torch.cat(yy).reshape(-1, 1)
        zz = torch.cat(zz).reshape(-1, 1)
        return torch.cat((xx, yy, zz), dim=1)
