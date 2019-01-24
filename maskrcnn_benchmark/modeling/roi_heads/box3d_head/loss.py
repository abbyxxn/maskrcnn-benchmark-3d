import torch
import torch.nn as nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.box3d_coder import Box3dCoder
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.orientation_coder import OrientationCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


class Box3DLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, box3d_coder, orientation_coder, num_bins):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.orientation_coder = orientation_coder
        self.box3d_coder = box3d_coder
        self.num_bins = num_bins
        self.confidence_loss = torch.nn.BCEWithLogitsLoss()
        self.orientation_loss = OrientationLoss()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "boxes_3d", "alphas"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets, img_original_ids):
        labels = []
        boxes3d_targets = []
        confidences = []
        orientations = []
        for proposals_per_image, targets_per_image, img_name in zip(proposals, targets, img_original_ids):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            bounding_box_3d = matched_targets.get_field("boxes_3d")
            bounding_box_3d = bounding_box_3d[positive_inds]

            alpha_orientation = matched_targets.get_field("alphas")
            alpha_orientation = alpha_orientation[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            assert bounding_box_3d.size == positive_proposals.size, "{}, {}".format(
                bounding_box_3d, positive_proposals
            )

            proposals_per_image = proposals_per_image.convert("xywh")
            confidences_per_image, orientations_per_image = self.orientation_coder.encode(alpha_orientation)
            bounding_box_3d_per_image = self.box3d_coder.encode(proposals_per_image.bbox,
                                                                bounding_box_3d.bbox_3d, labels_per_image, img_name
                                                                )
            proposals_per_image = proposals_per_image.convert("xyxy")
            confidences.append(confidences_per_image)
            orientations.append(orientations_per_image)
            labels.append(labels_per_image)
            boxes3d_targets.append(bounding_box_3d_per_image)

        return labels, boxes3d_targets, confidences, orientations

    def __call__(self, proposals, box3d_dim_regression,
                 box3d_rotation_logits,
                 box3d_rotation_regression,
                 box3d_localization_conv_regression,
                 box3d_localization_pc_regression,
                 targets, img_original_ids):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.
        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        device = box3d_dim_regression.device
        labels, boxes3d_targets, confidences_targets, orientations_targets = self.prepare_targets(proposals, targets,
                                                                                                  img_original_ids)
        labels = cat(labels, dim=0)
        boxes3d_targets = cat(boxes3d_targets, dim=0)
        confidences_targets = cat(confidences_targets, dim=0)
        orientations_targets = cat(orientations_targets, dim=0)

        labels = labels - 1
        map_inds = 3 * labels.cpu()[:, None] + torch.tensor([0, 1, 2])
        num_box3d = box3d_dim_regression.shape[0]
        index = torch.arange(num_box3d, device=device)

        boxes3d_targets = torch.as_tensor(boxes3d_targets, dtype=torch.float32, device=device)
        box3d_loss = smooth_l1_loss(
            box3d_dim_regression[index[:, None], map_inds],
            boxes3d_targets[:, 1:4],
            size_average=False,
            beta=1,
        )
        box3d_loss = box3d_loss / labels.numel()
        box3d_localization_loss = smooth_l1_loss(
            box3d_localization_conv_regression[index[:, None], map_inds] + box3d_localization_pc_regression[
                index[:, None], map_inds],
            boxes3d_targets[:, 4:7],
            size_average=False,
            beta=1,
        )
        box3d_localization_loss = box3d_localization_loss / labels.numel()

        confidences_targets = torch.as_tensor(confidences_targets, dtype=torch.float32, device=device)

        rotation_confidence_loss = self.confidence_loss(box3d_rotation_logits, confidences_targets)

        orientations_targets = torch.as_tensor(orientations_targets, dtype=torch.float32, device=device)
        box3d_rotation_regression = box3d_rotation_regression.reshape(-1, self.num_bins, 2)
        box3d_rotation_regression = F.normalize(box3d_rotation_regression, dim=2)
        rotation_regression_loss = self.orientation_loss(orientations_targets, box3d_rotation_regression)

        return box3d_loss, rotation_confidence_loss, rotation_regression_loss, box3d_localization_loss


class OrientationLoss(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()

    def forward(self, y_true, y_pred):
        device = y_true.device
        anchors = torch.sum(torch.mul(y_true, y_true), dim=2)
        anchors = anchors > 0.5
        anchors = torch.sum(torch.as_tensor(anchors, dtype=torch.float32, device=device), dim=1)

        loss = (y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
        loss = torch.sum(loss, dim=1)
        loss = loss / anchors
        loss = torch.mean(loss)
        loss = 2 - 2 * loss

        return loss


def make_roi_box3d_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox3d_reg_weights = cfg.MODEL.ROI_BOX3D_HEAD.BBOX3D_REG_WEIGHTS
    box3d_coder = Box3dCoder(weights=bbox3d_reg_weights)

    orientation_coder = OrientationCoder(cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_BIN,
                                         cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_OVERLAP)

    loss_evaluator = Box3DLossComputation(matcher, box3d_coder,
                                          orientation_coder, cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_BIN)
    return loss_evaluator
