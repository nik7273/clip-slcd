import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import torch.nn.functional as F



if __name__ == '__main__':
    datapath = '' #path to uhumans/left_images goes here
    path_norm = 'reconstructions/uhuman_apartment' # tum3d_room_norm'
    # path_extra = 'reconstructions/uhuman_apartment_extra/' #tum3d_room_extra'
    
    traj_gt = np.load(os.path.join(path_norm, 'poses_ref.npy'))
    traj_norm = np.load(os.path.join(path_norm, 'poses_est.npy'))
    # traj_extra = np.load(os.path.join(path_extra, 'poses_est_unop.npy'))
    # traj_extra_op = np.load(os.path.join(path_extra, 'poses_est.npy'))
    # time_norm_idx = np.load(os.path.join(path_norm, 'tstamps.npy'))
    # time_extra_idx = np.load(os.path.join(path_extra, 'tstamps.npy'))
    
    # image_path = os.path.join(datapath, 'rgb')
    # images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    # tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    # traj_est = PoseTrajectory3D(
    #     positions_xyz=traj_est[:,:3],
    #     orientations_quat_wxyz=traj_est[:,3:],
    #     timestamps=np.array(tstamps))
    
    # # FIXME: orignal way of loading
    # gt_file = os.path.join(datapath, 'groundtruth.txt')
    # traj_gt = file_interface.read_tum_trajectory_file(gt_file)
    
    # T = traj_gt[0]
    # Rot_mat = R.from_quat([quat[0, 0], quat[0, 1], quat[0, 2], quat[0, 3]]).as_matrix()
    # trans = traj_gt[0, :3].reshape(1, -1)
    
    # if True:
    #     Rot_mat = Rot_mat.T
    #     trans = - trans @ Rot_mat.T
    
    
    # traj_norm[:, :3] = traj_norm[:, :3] @ Rot_mat.T + trans
    # traj_extra[:, :3] = traj_extra[:, :3] @ Rot_mat.T + trans
    
    gt_range = np.max(traj_gt, axis=0) - np.min(traj_gt, axis=0)
    norm_range = np.max(traj_norm, axis=0) - np.min(traj_norm, axis=0)
    # extra_range = np.max(traj_extra, axis=0) - np.min(traj_extra, axis=0)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(traj_norm[:, 0], traj_norm[:, 1], traj_norm[:, 2], color='r', marker='o', label='Prediction without extra edge')
    # ax.plot(traj_extra[:, 0], traj_extra[:, 1], traj_extra[:, 2], color='b', marker='o', label='Prediction with extra edge')
    # ax.plot(traj_extra_op[:, 0], traj_extra_op[:, 1], traj_extra_op[:, 2], color='m', marker='o', label='Prediction with extra edge (optimize)')
    # ax.plot3D(traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2], color='g', marker='o', label='Ground Truth')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    # # 2d plot
    # plt.plot(traj_norm[:, 0], traj_norm[:, 1], color='r', marker='o', label='Prediction without extra edge')
    # plt.plot(traj_extra[:, 0], traj_extra[:, 1], color='b', marker='o', label='Prediction with extra edge')
    plt.plot(traj_gt[:, 0], traj_gt[:, 1], color='g', marker='o', label='Groud truth')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')

    # ax.set_aspect('equal')
    
    ax.legend()
    plt.show()
    
