"""
Simple dataset class that wraps a list of path names
"""

import _pickle as cPickle
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box_3d import Box3List

from maskrcnn_benchmark.structures.bounding_box import BoxList

TYPICAL_DIMENSION = {}


class KITTIDataset(data.Dataset):
    def __init__(self, root, ann_file, remove_images_without_annotations, transforms=None):
        super(KITTIDataset, self).__init__()
        self.root = root
        self.image_name, self.label_list, self.boxes_list, self.boxes_3d_list, self.alphas_list = self.get_pkl_element(
            ann_file)
        number_image = len(self.image_name)
        self.image_lists = []
        self.calib_lists = []
        self.depth_list = []
        self.pseudo_pc_list = []
        for i in range(number_image):
            self.image_lists.append(root + '/training' + '/image_2/' + self.image_name[i] + ".png")
            self.calib_lists.append(root + '/training' + '/calib/' + self.image_name[i] + ".txt")
            self.depth_list.append(root + '/training' + '/depth/' + self.image_name[i] + ".npz")
            self.pseudo_pc_list.append(root + '/training' + '/pseudo_pc/' + self.image_name[i] + ".npz")
        self.ids = list(range(number_image))
        self.transforms = transforms

        self.category_id_to_label_name = {
            1: "Car",
        }
        self.label_name_to_category_id = {
            "Car": 1,
        }

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.boxes_list[img_id]) > 0
            ]

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

    def __getitem__(self, index):
        idx = self.id_to_img_map[index]
        img = Image.open(self.image_lists[idx]).convert("RGB")
        boxes = self.boxes_list[idx]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = self.label_list[idx]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        boxes_3d = self.boxes_3d_list[idx]
        boxes_3d = torch.as_tensor(boxes_3d).reshape(-1, 7)
        boxes_3d = Box3List(boxes_3d, img.size)
        target.add_field("boxes_3d", boxes_3d)

        alphas = self.alphas_list[idx]
        alphas = torch.tensor(alphas)
        target.add_field("alphas", alphas)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        img_original_idx = self.image_name[idx]

        return img, target, index, img_original_idx

    def __len__(self):
        return len(self.ids)

    def get_alpha(self, calib_lists, boxes_list):
        alpha_list = []
        for i, filename in enumerate(calib_lists):
            with open(filename, 'r') as f:
                calib = {}
                for line in f:
                    fields = line.split()
                    if len(fields) is 0:
                        continue
                    key = fields[0][:-1]
                    val = np.asmatrix(fields[1:]).astype(np.float32).reshape(3, -1)
                    calib[key] = val
                fx = calib['P2'][0, 0]
                cx = calib['P2'][0, 2]
                x = (boxes_list[i][:, 0] + boxes_list[i][:, 2]) / 2
                alpha_list.append(np.arctan2(x - cx, fx))
        return alpha_list

    def get_img_info(self, index):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        idx = self.id_to_img_map[index]
        img = Image.open(self.image_lists[idx]).convert("RGB")
        width, height = img.size
        return {"height": height, "width": width}

    @staticmethod
    def get_pkl_element(ann_file):
        '''
        labels mapping:
        1 : person_sitting, pedestrian  pedestrian
        2 : cyclist, riding             cyclist
        3 : van, car                    car
        4 : train, bus
        5 : truck
        :param ann_file:
        :return:
        '''
        image_name = {}
        labels = {}
        boxes_list = {}
        boxes_3d_list = {}
        alphas_list = {}
        index = 0
        if os.path.exists(ann_file):
            with open(ann_file, 'rb') as file:
                roidb = cPickle.load(file)
                for roi in roidb:
                    image_name[index] = roi['image_original_index']
                    labels[index] = roi['label']
                    boxes_list[index] = roi['boxes']
                    boxes_3d_list[index] = roi['boxes_3d']
                    alphas_list[index] = roi['alphas']
                    index = index + 1
        return image_name, labels, boxes_list, boxes_3d_list, alphas_list

    @staticmethod
    def get_typical_dimension(label_list, boxes_3d_list):
        typical_dimension = {}
        categories = {}
        for index, label in label_list.items():
            for i, boxes_3d in enumerate(boxes_3d_list[index]):
                value = typical_dimension.get(label[i], [0, 0, 0])
                count = categories.get(label[i], 0)
                value = value + boxes_3d[1:4]
                count = count + 1
                typical_dimension[label[i]] = value
                categories[label[i]] = count
        result = {}
        for k, v in typical_dimension.items():
            result[k] = v / categories[k]

        return result  # lhw
