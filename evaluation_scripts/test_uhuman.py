import sys

from MixVPR.main import VPRModel
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import torch.nn.functional as F
import torchvision
from droid import Droid
import networkx as nx
import matplotlib.pyplot as plt


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, image_size=[320, 512], stride = 2):
    """ image generator """

    fx, fy, cx, cy = 415.69219381653056, 415.69219381653056, 360.0, 240.0

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.0, 0.0, 0.0, 0.0])

    # read all png images in folder
    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::stride]
    
    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)
        ht0, wd0, _ = image.shape
        image = cv2.undistort(image, K_l, d_l)
        image = cv2.resize(image, (320+32, 240+16))
        image = torch.from_numpy(image).permute(2,0,1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda()
        intrinsics[0] *= image.shape[2] / 640.0
        intrinsics[1] *= image.shape[1] / 480.0
        intrinsics[2] *= image.shape[2] / 640.0
        intrinsics[3] *= image.shape[1] / 480.0

        # crop image to remove distortion boundary
        intrinsics[2] -= 16
        intrinsics[3] -= 8
        image = image[:, 8:-8, 16:-16]

        yield t, image[None], intrinsics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default='/media/jimmy/TOSHIBA EXT/Data_Set/uhumans/apartment_scene/left_images')
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.25)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)
    stride = 2
    droid = Droid(args)
    time.sleep(5)

    lcd_range = 50 #range beyond which loop closure is considered 

    
    tstamps = []
    similar_indices = []
    i = 0
    for (t, image, intrinsics) in tqdm(image_stream(args.datapath)):
        
        # if not args.disable_vis:
        #     show_image(image)
        droid.track(t, image, intrinsics=intrinsics)
        print(t)
        # print(droid.video.tstamp)
        # print(droid.video.counter)
        # print(torch.any(i == droid.video.tstamp))

        #check if the current frame is a keyframe
        

            # 00000000000|12
            # 12345677891|000
        # if i > 100: 
        #     break
        
        #check if i is in the tensor droid.video.tstamp 
        
        # print(f"Shape: {image.shape}\nType: {type(image)}")
        
        i += 1
        

    #create a dict with the keys as the indices and the values as the elements of similar_indices 
    d = {}
    for i in range(len(similar_indices)):
        d[i] = similar_indices[i]

    

    traj_est = droid.terminate(image_stream(args.datapath))

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, 'rgb')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::stride]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

    np.save("reconstructions/uhuman_apartment/traj_est.npy", traj_est.positions_xyz)
    gt_file = os.path.join(args.datapath, 'poses.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    G = nx.Graph(droid.loop_candidates)
    pos = {}
    for i in range(traj_est.positions_xyz.shape[0]): 
        pose = traj_est.positions_xyz[i]
        pos[i] = (pose[2], pose[0])#, pose[2])
    nx.draw(G, pos, with_labels=True)
    plt.show()

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    # result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
    #     pose_relation=PoseRelation.translation_part, align=True, correct_scale=True, save_path=args.reconstruction_path)
    # result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
    #     pose_relation=PoseRelation.full_transformation, align=True, correct_scale=True, save_path=args.reconstruction_path)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True, save_path=args.reconstruction_path)


    print(result)
