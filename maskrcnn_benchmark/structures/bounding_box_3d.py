# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Box3dList(object):
    """
    This class represents a set of 3D bounding boxes.
    The bounding boxes are represented as a Nx7 Tensor.
    In order ot uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox_3d, image_size, mode="ry-lhwxyz"):
        device = bbox_3d.device if isinstance(bbox_3d, torch.Tensor) else torch.device("cpu")
        bbox_3d = torch.as_tensor(bbox_3d, dtype=torch.float32, device=device)
        if bbox_3d.ndimension() != 2:
            raise ValueError(
                "bbox_3d should have 2 dimensions, got {} {}".format(bbox_3d.ndimension())
            )
        if bbox_3d.size(-1) != 7:
            raise ValueError(
                "last dimenion of bbox_3d should have a "
                "size of 7, got {}".format(bbox_3d.size(-1))
            )
        if mode not in ("ry-lhwxyz",):
            raise ValueError("mode should be 'ry-lhwxyz'")

        self.bbox_3d = bbox_3d
        self.rotation = [BoundingBox3DRotation(r, image_size, mode) for r in bbox_3d[:, :1]]
        self.dimension = [BoundingBox3DDimension(d, image_size, mode) for d in bbox_3d[:, 1:4]]
        self.localization = [BoundingBox3DLocalization(l, image_size, mode) for l in bbox_3d[:, 4:]]
        self.size = image_size  # (image_width, image_height)
        self.mode = mode

    def convert(self, mode):
        if mode not in ("ry-lhwxyz"):
            raise NotImplementedError(
                "Only ry-lhwxyz mode implemented"
            )

    def _split_into_ry_lhwxyz(self):
        if self.mode == "ry-lhwxyz":
            rotation, length, height, width, x, y, z = self.bbox_3d.split(1, dim=-1)
            return rotation, length, height, width, x, y, z
        else:
            raise RuntimeError("Only ry-lhwxyz mode implemented")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox_3d * ratio
            bbox_3d = Box3List(scaled_box, size, mode=self.mode)
            # bbox_3d._copy_extra_fields(self)
            return bbox_3d

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox_3d = BoxList(scaled_box, size, mode="xyxy")
        # bbox_3d._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox_3d.add_field(k, v)

        return bbox_3d.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        rotation, length, height, width, x, y, z = self._split_into_ry_lhwxyz()
        if method == FLIP_LEFT_RIGHT:
            PI = 3.14
            transposed_rotation = rotation
            for i, rotation_item in enumerate(rotation):
                if rotation_item > 0:
                    rotation_item = PI - rotation_item
                elif rotation_item < 0:
                    rotation_item = -PI - rotation_item
                transposed_rotation[i] = rotation_item
            transposed_length = length
            transposed_height = height
            transposed_width = width
            transposed_x = -x
            transposed_y = y
            transposed_z = z
        elif method == FLIP_TOP_BOTTOM:
            transposed_rotation = rotation
            transposed_length = length
            transposed_height = height
            transposed_width = width
            transposed_x = x
            transposed_y = -y
            transposed_z = z
        transposed_boxes_3d = torch.cat(
            (transposed_rotation, transposed_length, transposed_height, transposed_width, transposed_x, transposed_y,
             transposed_z), dim=-1
        )
        bbox_3d = Box3List(transposed_boxes_3d, self.size, mode="ry-lhwxyz")
        # bbox_3d._copy_extra_fields(self)
        return bbox_3d

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox_3d = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox_3d._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox_3d.add_field(k, v)
        return bbox_3d.convert(self.mode)


    def to(self, device):
        bbox_3d = Box3List(self.bbox_3d.to(device), self.size, self.mode)
        return bbox_3d

    def __getitem__(self, item):
        bbox_3d = Box3List(self.bbox_3d[item], self.size, self.mode)
        return bbox_3d

    def __len__(self):
        return self.bbox_3d.shape[0]

    # TODO resize, crop, clip_to_image,

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox_3d[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox_3d[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox_3d[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox_3d[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox_3d
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        TO_REMOVE = 1
        box = self.bbox_3d
        area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        return area

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


class BoundingBox3DLocalization(object):
    def __init__(self, localization, size, mode):
        if isinstance(localization, list):
            localization = [torch.as_tensor(l, dtype=torch.float32) for l in localization]

        self.localization = localization
        self.size = size
        self.mode = mode


class BoundingBox3DDimension(object):
    def __init__(self, dimension, size, mode):
        if isinstance(dimension, list):
            dimension = [torch.as_tensor(d, dtype=torch.float32) for d in dimension]

        self.dimension = dimension
        self.size = size
        self.mode = mode


class BoundingBox3DRotation(object):
    def __init__(self, rotation, size, mode):
        if isinstance(rotation, list):
            rotation = [torch.as_tensor(r, dtype=torch.float32) for r in rotation]

        self.rotation = rotation
        self.size = size
        self.mode = mode

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes3d={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox_3d = Box3List([[0, 0, 10, 10, 10, 10, 0], [0, 0, 5, 5, 5, 5, 0]], (10, 10))
    s_bbox_3d = bbox_3d.resize((5, 5))
    print(s_bbox_3d)
    print(s_bbox_3d.bbox_3d)

    t_bbox_3d = bbox_3d.transpose(0)
    print(t_bbox_3d)
    print(t_bbox_3d.bbox_3d)
