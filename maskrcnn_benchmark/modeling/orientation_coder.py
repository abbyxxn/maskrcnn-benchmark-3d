import math

import numpy as np
import torch
import torch.nn.functional as F

PI = 3.14159


class OrientationCoder(object):
    """
    This class encodes and decodes a set of 3d bounding boxes orientation into
    the representation used for training the regressors.
    """

    def __init__(self, num_bins, overlap):
        """
        Arguments:
            num_bins (int)
            overlap (float)
        """
        self.num_bins = num_bins
        self.overlap = overlap

    def encode(self, reference_orientation):
        """
        Encode a set of reference orientation

        Arguments:
            reference_orientation (Tensor): reference boxes
        """
        if len(reference_orientation) == 0:
            return torch.empty(0, dtype=torch.float32)
        orientations = torch.zeros(reference_orientation.shape[0], self.num_bins, 2)
        confidences = torch.zeros(reference_orientation.shape[0], self.num_bins)
        for i, alpha in enumerate(reference_orientation):
            orientation = torch.zeros(self.num_bins, 2)
            confidence = torch.zeros(self.num_bins)
            anchors = self.compute_anchors(alpha)
            for anchor in anchors:
                orientation[anchor[0], :] = torch.tensor([torch.cos(anchor[1]), torch.sin(anchor[1])])
                confidence[anchor[0]] = 1.
            orientations[i, :, :] = orientation
            confidences[i, :] = confidence
        return confidences, orientations

    def angle_from_multibin(angle_conf, angle_loc, overlap):
        num_bins = angle_conf.shape[1]
        bins = np.zeros((num_bins, 2), dtype=np.float32)
        bin_angle = 2 * PI / num_bins + overlap
        start = -PI - overlap / 2
        for i in range(num_bins):
            bins[i, 0] = start
            bins[i, 1] = start + bin_angle
            start = bins[i, 1] - overlap

        alphas = np.zeros((angle_conf.shape[0],), dtype=np.float32)

        for k in range(angle_conf.shape[0]):
            bin_ctrs = ((bins[:, 0] + bins[:, 1]) / 2).reshape(1, -1)  # 1 x num_bins
            conf_ctr = bin_ctrs[0, np.argmax(angle_conf[k, :].reshape(1, -1))]
            ind = np.argmax(angle_conf[k, :])
            cos_alpha = angle_loc[k, 2 * ind]
            sin_alpha = angle_loc[k, 2 * ind + 1]
            loc_alpha = np.arctan2(sin_alpha, cos_alpha)
            alphas[k] = conf_ctr + loc_alpha

        return alphas

    def compute_anchors(self, alpha):
        alpha = alpha + PI / 2
        if alpha < 0:
            alpha = alpha + 2. * PI
        alpha = alpha - int(alpha / (2. * PI)) * (2. * PI)

        anchors = []
        wedge = 2. * PI / self.num_bins

        l_index = int(alpha / wedge)
        r_index = l_index + 1

        if (alpha - l_index * wedge) < wedge / 2 * (1 + self.overlap / 2):
            anchors.append([l_index, alpha - l_index * wedge])

        if (r_index * wedge - alpha) < wedge / 2 * (1 + self.overlap / 2):
            anchors.append([r_index % self.num_bins, alpha - r_index * wedge])

        return anchors

    def decode(self, box3d_rotation_logits, box3d_rotation_regression):
        """
        From a set of orientation_confidences and encoded relative orientation offsets,
        get the decoded box orientation.

        Arguments:
            orientation_offset (Tensor): encoded orientation offset
            orientation_confidence (Tensor): the max confidence bin
        """
        device = box3d_rotation_logits.device
        bin_prob = F.softmax(box3d_rotation_logits, -1)
        max_bin = torch.argmax(bin_prob, dim=1)
        box3d_rotation_regression = box3d_rotation_regression.reshape(-1, self.num_bins, 2)
        index = torch.arange(box3d_rotation_logits.shape[0])
        anchors = box3d_rotation_regression[index, max_bin, :]
        angle_offsets = torch.zeros((box3d_rotation_logits.shape[0]), device=device)
        for i, anchor in enumerate(anchors):
            if anchor[1] > 0:
                acos_offset = math.modf(anchor[0])[0]
                acos_offset = torch.as_tensor(acos_offset, dtype=torch.float32, device=device)
                angle_offset = torch.acos(acos_offset)
                angle_offsets[i] = angle_offset
            else:
                acos_offset = math.modf(anchor[0])[0]
                acos_offset = torch.as_tensor(acos_offset, dtype=torch.float32, device=device)
                angle_offset = -torch.acos(acos_offset)
                angle_offsets[i] = angle_offset

        wedge = 2. * PI / self.num_bins
        max_bin = torch.as_tensor(max_bin, dtype=torch.float32, device=device)
        angle_offsets = angle_offsets + max_bin * wedge
        angle_offsets = angle_offsets % (2. * PI)

        angle_offsets = angle_offsets - PI / 2
        for i, angle_offset in enumerate(angle_offsets):
            if angle_offset > PI:
                angle_offset = angle_offset - (2. * PI)
                angle_offsets[i] = angle_offset
            else:
                angle_offsets[i] = angle_offset

        return angle_offsets
