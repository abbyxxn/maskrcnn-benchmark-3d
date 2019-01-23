import _pickle as cPickle
import json
import os

import torch
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


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
    image_original_index = {}
    labels = {}
    boxes_list = {}
    boxes_3d_list = {}
    alphas_list = {}
    index = 0
    if os.path.exists(ann_file):
        with open(ann_file, 'rb') as file:
            roidb = cPickle.load(file)
            for roi in roidb:
                image_original_index[index] = roi['image_original_index']
                labels[index] = roi['label']
                boxes_list[index] = roi['boxes']
                boxes_3d_list[index] = roi['boxes_3d']
                alphas_list[index] = roi['alphas']
                index = index + 1
    return image_original_index, labels, boxes_list, boxes_3d_list, alphas_list


def convert_kitti_instance_only(root, ann_file, out_dir, dataset):
    image_index, label_list, boxes_list, boxes_3d_list, \
    alphas_list = get_pkl_element(ann_file)
    number_image = len(image_index)
    image_lists = []
    calib_lists = []
    depth_list = []
    for i in range(number_image):
        image_lists.append(root + '/training' + '/image_2/' + image_index[i] + ".png")
        calib_lists.append(root + '/training' + '/calib/' + image_index[i] + ".txt")
        depth_list.append(root + '/training' + '/depth/' + image_index[i] + "_01.png.npz")

    # img_id = 0
    # ann_id = 0
    img_id = 3712
    ann_id = 11855

    # cat_id = 1
    category_dict = {'car': 1}

    category_instancesonly = [
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    ]

    ann_dict = {}
    images = []
    annotations = []

    for i, id in image_index.items():
        if len(images) % 50 == 0:
            print("Processed %s images, %s annotations" % (
                len(images), len(annotations)))
        image = {}
        image['id'] = img_id
        img_id += 1

        img = Image.open(image_lists[i]).convert("RGB")
        width, height = img.size
        image['width'] = width
        image['height'] = height
        image['file_name'] = image_lists[i].split('/')[-1]
        image['seg_file_name'] = image['file_name']

        images.append(image)

        num_instances = label_list[i].shape[0]
        boxes = boxes_list[i]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        box2d = BoxList(boxes, img.size, mode="xyxy")
        area = box2d.area().tolist()
        boxes = box2d.convert('xywh')
        boxes = boxes.bbox.tolist()

        for j in range(num_instances):
            ann = {}
            ann['id'] = ann_id
            ann_id += 1
            ann['image_id'] = image['id']
            ann['segmentation'] = []

            ann['category_id'] = category_dict['car']
            ann['iscrowd'] = 0
            ann['area'] = area[j]
            ann['bbox'] = boxes[j]

            annotations.append(ann)

    ann_dict['images'] = images
    categories = [{"id": category_dict[name], "name": name} for name in
                  category_dict]
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))

    with open(
            os.path.join(out_dir, 'instancesonly_filtered_gtFine_' + dataset + '.json'),
            'w') as outfile:
        outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    root = '/home/abby/Repositories/maskrcnn-benchmark/datasets/kitti/object'
    ann_file = '/home/abby/raid/dataset/kitti/object/kitti_val_car_gt_roidb.pkl'
    output_dir = '/home/abby/Repositories/maskrcnn-benchmark/datasets/kitti/object'
    dataset = 'val'  # train or val or test
    convert_kitti_instance_only(root, ann_file, output_dir, dataset)
