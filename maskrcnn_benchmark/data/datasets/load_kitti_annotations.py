import _pickle as cPickle
import os

import numpy as np


class KITTIObject(object):
    def __init__(self, image_set, dataset_path, split_dataset=False, split_factor=0.8):
        self.dataset_path = dataset_path

        self.name = 'kitti_' + image_set
        self._image_set = image_set
        self._subset = image_set.split('_')[-1]  # car or ped_cyc
        self._dataset = image_set.split('_')[0]  # train / val /  test
        self._data_path = self._get_default_data_path()
        self.split_dataset = split_dataset
        self.split_factor = split_factor

        if self._subset == 'car':
            self.classes = ('__background__',  # always index 0
                            'car')
            self.num_classes = 2
        if self._subset == 'vehicle':
            self.classes = ('__background__',  # always index 0
                            'vehicle')
            self.num_classes = 2
        elif self._subset == 'ped_cyc':
            self.classes = ('__background__',  # always index 0
                            'pedestrian', 'cyclist')
            self.num_classes = 3
        elif self._subset == 'all':
            # self.classes = ('__background__', # always index 0
            #                  'car', 'pedestrian', 'cyclist')
            self.classes = ('__background__',  # always index 0
                            'pedestrian', 'riding', 'car', 'bus',
                            'truck')
            self.num_classes = len(self.classes)
        # assert self.num_classes >= 2, 'ERROR: incorrect subset'
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self._image_ext = '.png'
        if self.split_dataset:
            self._image_index = self._load_image_set_index()
            self.validation_image_set_index, self.training_image_set_index = self.split_dataset_function(
                image_index=self._image_index, factor=self.split_factor)
            # TODO split dataset
        else:
            self._image_index = self._load_image_set_index()
        self.num_images = len(self._image_index)
        self.competition_mode(False)
        self.cls_stats = dict()
        for fg_cls in self.classes[1:]:
            self.cls_stats[fg_cls] = 0

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000,
                       'rpn_file': None}

        self._gt_splits = ('train', 'val', 'trainval', 'minval')
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
        self.cache_path = self.dataset_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cls_stats_cache = {}
        for key in self.cls_stats:
            cls_stats_cache[key] = 0
        if self.split_dataset:
            cache_file = os.path.join(self.cache_path, 'kitti_validation_' + self.name.split('_')[-1] + '_gt_roidb.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb_val = cPickle.load(fid)
                print('kitti validation gt roidb loaded from {}'.format(cache_file))
            else:
                print('Load kitti validation annotations...')
                roidb_val = [self._load_kitti_annotation(index)
                             for index in self.validation_image_set_index]
                print('RoIdb stats (class/numbers):')
                print(self.cls_stats)
                with open(cache_file, 'wb') as fid:
                    cPickle.dump(roidb_val, fid)
                print('wrote validation gt roidb to {}'.format(cache_file))

            self.cls_stats = cls_stats_cache
            cache_file = os.path.join(self.cache_path, 'kitti_train_' + self.name.split('_')[-1] + '_gt_roidb.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb_train = cPickle.load(fid)
                print('kitti train gt roidb loaded from {}'.format(cache_file))
            else:
                print('Load kitti train annotations...')
                roidb_train = [self._load_kitti_annotation(index)
                               for index in self.training_image_set_index]
                print('RoIdb stats (class/numbers):')
                print(self.cls_stats)
                with open(cache_file, 'wb') as fid:
                    cPickle.dump(roidb_train, fid)
                print('wrote train gt roidb to {}'.format(cache_file))
        else:
            cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb = cPickle.load(fid)
                print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            else:
                print('Load kitti annotations...')
                roidb = [self._load_kitti_annotation(index)
                         for index in self._image_index]
                print('RoIdb stats (class/numbers):')
                print(self.cls_stats)
                with open(cache_file, 'wb') as fid:
                    cPickle.dump(roidb, fid)
                print('wrote gt roidb to {}'.format(cache_file))

        return

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'image_2',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        image_set_file = os.path.join(self.dataset_path, "ImageSets", self._dataset + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def split_dataset_function(self, image_index, factor=0.8):
        image_ind = []
        for image in image_index:
            image_ind.append(image.split('_')[0])
        np.random.seed()
        np.random.shuffle(image_ind)
        train_count = (len(image_ind) * int(factor * 100)) // 100
        train = image_ind[:train_count]
        val = image_ind[train_count:]
        return val, train

    def _get_default_data_path(self):
        """
        Return the default data path where PASCAL VOC is expected to be installed.
        """
        if self._dataset in ['train', 'val', 'trainval', 'minval']:
            return os.path.join(self.dataset_path, 'training')
        elif self._dataset in ['test']:
            return os.path.join(self.dataset_path, 'testing')

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from kitti annotationl
        format.
        """
        index = index.split('_')[0]
        filename = os.path.join(self._data_path, 'label_2', index + '.txt')

        with open(filename, 'r') as f:
            lines = f.readlines()
        num_objs = len(lines)
        label = np.zeros((num_objs), dtype=np.int32)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((num_objs, 7), dtype=np.float32)
        alphas = np.zeros((num_objs), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        ignored = np.zeros((num_objs), dtype=np.bool)

        # Load object bounding boxes into a data frame.
        ix = -1
        for line in lines:
            obj = line.strip().split(' ')
            try:
                if self._subset == 'vehicle':
                    if obj[0].lower().strip() in ['car', 'van', 'truck', 'tram']:
                        cls_str = 'vehicle'
                    else:
                        cls_str = obj[0].lower().strip()
                    cls = self._class_to_ind[cls_str]
                elif self._subset == 'all':
                    cls_raw = obj[0].lower().strip()
                    if cls_raw in ['van']:
                        cls_str = 'car'
                    elif cls_raw in ['cyclist']:
                        cls_str = 'riding'
                    elif cls_raw in ['person_sitting', ]:
                        cls_str = 'pedestrian'
                    elif cls_raw in ['tram']:
                        cls_str = 'bus'
                    else:
                        cls_str = cls_raw
                    cls = self._class_to_ind[cls_str]
                elif self._subset == 'car':
                    if obj[0].lower().strip() in ['van']:
                        cls_str = 'car'
                    else:
                        cls_str = obj[0].lower().strip()
                    cls = self._class_to_ind[cls_str]
                else:
                    cls = self._class_to_ind[obj[0].lower().strip()]
            except:
                continue
            # ignore objects with undetermined difficult level
            level = self._get_obj_level(obj)
            if level > 3:
                continue
            # if level != 3:
            #     continue
            if cls_str in self.classes:
                self.cls_stats[cls_str] += 1
            ix += 1
            # 0-based coordinates
            alpha = float(obj[3])
            x1 = float(obj[4])
            y1 = float(obj[5])
            x2 = float(obj[6])
            y2 = float(obj[7])
            ry = float(obj[14])
            l = float(obj[10])
            h = float(obj[8])
            w = float(obj[9])
            tx = float(obj[11])
            ty = float(obj[12])
            tz = float(obj[13])
            # print(l,h,w,cls_raw)
            label[ix] = cls
            alphas[ix] = alpha
            boxes[ix, :] = [x1, y1, x2, y2]
            boxes_3d[ix, :] = [ry, l, h, w, tx, ty, tz]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            ignored[ix] = 0
        label = label[:ix + 1]
        alphas = alphas[:ix + 1]
        boxes = boxes[:ix + 1, :]
        boxes_3d = boxes_3d[:ix + 1, :]
        gt_classes = gt_classes[:ix + 1]
        ignored = ignored[:ix + 1]
        overlaps = overlaps[:ix + 1, :]

        return {'image_original_index': index,
                'label': label,
                'boxes': boxes,
                'boxes_3d': boxes_3d,
                # 'velo_xyz': velo_xyz,
                'alphas': alphas,
                }

    def _get_obj_level(self, obj):
        height = float(obj[7]) - float(obj[5]) + 1
        trucation = float(obj[1])
        occlusion = float(obj[2])
        if height >= 40 and trucation <= 0.15 and occlusion <= 0:
            return 1
        elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
            return 2
        elif height >= 25 and trucation <= 0.5 and occlusion <= 2:
            return 3
        else:
            return 4

    def competition_mode(self, on):
        if on:
            pass

        else:
            pass


if __name__ == '__main__':
    d = KITTIObject('train_car', '/home/abby/raid/dataset/kitti/object', split_dataset=False, split_factor=0.999)
    d.gt_roidb()
    print('roi')
