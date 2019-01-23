import os

import cv2
import numpy as np
from scipy import linalg


class Camera(object):
    """ Class for representing pin-hole cameras. """

    def __init__(self, P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K, self.R, self.t = self.factor()

    @property
    def f(self):
        return self.K[0, 0]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def tx(self):
        return self.t[0, 0]

    @property
    def ty(self):
        return self.t[1, 0]

    def factor(self):
        """ Factorize the camera matrix into K,R,t as P = K[R|t]. """

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1, 1] *= -1

        K = np.dot(K, T)
        R = np.dot(T, R)  # T is its own inverse
        t = np.dot(linalg.inv(K), self.P[:, 3])

        return K, R, t


class Calib(object):
    def __init__(self, calib_path):
        calib = {}
        with open(calib_path, 'r') as cf:
            for line in cf:
                fields = line.split()
                if len(fields) is 0:
                    continue
                key = fields[0][:-1]
                val = np.asmatrix(fields[1:]).astype(np.float32).reshape(3, -1)
                calib[key] = val

        calib['Tr_velo_to_cam'] = np.vstack((calib['Tr_velo_to_cam'],
                                             [0, 0, 0, 1]))
        calib['R0_rect'] = np.hstack((calib['R0_rect'],
                                      np.zeros((3, 1))))
        calib['R0_rect'] = np.vstack((calib['R0_rect'],
                                      [0, 0, 0, 1]))
        calib['Tr_velo_to_rect'] = calib['R0_rect'] * calib['Tr_velo_to_cam']
        calib['Tr_rect_to_velo'] = np.linalg.inv(calib['Tr_velo_to_rect'])
        calib['P_velo_to_left'] = calib['P2'] * calib['Tr_velo_to_rect']
        calib['P_left'] = calib['P2']
        calib['P_right'] = calib['P3']
        self.calib = calib
        self.lcam = Camera(self.P2)
        self.rcam = Camera(self.P3)
        self.baseline = abs(self.lcam.tx - self.rcam.tx)

    def to_dict(self):
        return self.calib

    @property
    def Tr_velo_to_cam(self):
        return self.calib['Tr_velo_to_cam']

    @property
    def Tr_velo_to_rect(self):
        return self.calib['Tr_velo_to_rect']

    @property
    def Tr_rect_to_velo(self):
        return self.calib['Tr_rect_to_velo']

    @property
    def R0_rect(self):
        return self.calib['R0_rect']

    @property
    def P2(self):
        return self.calib['P2']

    @property
    def P3(self):
        return self.calib['P3']

    @property
    def P_velo_to_left(self):
        return self.calib['P_velo_to_left']

    '''
    Stereo helpers
    '''

    def load_disparity(self, disp_file):
        """
        load disparity from PNG file
        """
        # load uint16 PNG file
        disp = cv2.imread(disp_file, cv2.IMREAD_UNCHANGED)
        disp = disp.astype(np.float32) / 256.0
        disp[disp == 0] = 0.1

        return disp

    def pointcloud_from_disparity(self, disp, calib, lidar_coord=False):
        """
        compute depth from disparity
        """
        disp = disp.astype(np.float32)

        depth = calib.lcam.f * calib.baseline / disp
        height, width = depth.shape[0], depth.shape[1]
        original_depth = depth

        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        depth = depth.ravel()
        x = (u.ravel() - calib.lcam.cx) * depth / calib.lcam.f
        y = (v.ravel() - calib.lcam.cy) * depth / calib.lcam.f
        xyz = np.vstack((x, y, depth)).transpose((1, 0)).astype(np.float32)

        xyz = xyz.reshape(height, width, 3)
        xyz = np.moveaxis(xyz, -1, 0)

        return xyz

    def depth_from_disparity(self, disp, calib, lidar_coord=False):
        """
        compute depth from disparity
        """
        disp = disp.astype(np.float32)

        depth = calib.lcam.f * calib.baseline / disp

        return depth

    def load_stereo_pointcloud(self, disp_file, calib, lidar_coord=False):
        """
        load disparity and convert to point cloud using given camera parameters
        """
        # load uint16 PNG file
        disp = self.load_disparity(disp_file)
        xyz = self.pointcloud_from_disparity(disp, calib, lidar_coord)

        return xyz

    def load_stereo_depth(self, disp_file, calib, lidar_coord=False):
        """
        load disparity and convert to point cloud using given camera parameters
        """
        # load uint16 PNG file
        disp = self.load_disparity(disp_file)
        depth = self.depth_from_disparity(disp, calib, lidar_coord)

        return depth


if __name__ == "__main__":
    calib_dir = "/home/abby/datasets/kitti/object/training/calib"
    disparity_dir = "/home/abby/datasets/kitti/object/output_disparity"
    output_pc_dir = "/home/abby/datasets/kitti/object/pseudo_pc"  # xyz
    output_depth_dir = "/home/abby/datasets/kitti/object/depth"
    disparity_index = [img.split('.')[0] for img in os.listdir(calib_dir)]
    for disp_index in disparity_index:
        calib = Calib(os.path.join(calib_dir, disp_index + '.txt'))
        xyz = calib.load_stereo_pointcloud(os.path.join(disparity_dir, disp_index + '.png'), calib)
        np.savez_compressed(os.path.join(output_pc_dir, disp_index), pseudo_pc=xyz)
        # vis_pc.vis_in_ply(xyz, save_path='/home/abby/Repositories/ply-vis/')
        depth = calib.load_stereo_depth(os.path.join(disparity_dir, disp_index + '.png'), calib)
        np.savez_compressed(os.path.join(output_depth_dir, disp_index), depth=depth)
