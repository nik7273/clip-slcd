import torch
import lietorch
import numpy as np
from MixVPR.main import VPRModel

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video, datapath=self.args.datapath)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

        # self.load_clipvpr_weights()
    
    def load_clipvpr_weights(self):
        self.clipvpr_encoder = VPRModel(
            #---- Encoder
            backbone_arch='resnet50',
            pretrained=True,
            layers_to_freeze=2,
            layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
            
            #---- Aggregator
            # agg_arch='CosPlace',
            # agg_config={'in_dim': 2048,
            #             'out_dim': 2048},
            # agg_arch='GeM',
            # agg_config={'p': 3},
            
            # agg_arch='ConvAP',
            # agg_config={'in_channels': 2048,
            #             'out_channels': 2048},

            agg_arch='MixVPR',
            agg_config={'in_channels' : 2048, #change this to 1024 if no clip, but 2048 with clip 
                    'in_h' : 20,
                    'in_w' : 20,
                    'out_channels' : 1024,
                    'mix_depth' : 4,
                    'mlp_ratio' : 1,
                    'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
            
            #---- Train hyperparameters
            lr=0.05, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
            optimizer='sgd', # sgd, adamw
            weight_decay=0.001, # 0.001 for sgd and 0 for adam,
            momentum=0.9,
            warmpup_steps=650,
            milestones=[5, 10, 15, 25, 45],
            lr_mult=0.3,

            #----- Loss functions
            # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
            # FastAPLoss, CircleLoss, SupConLoss,
            loss_name='MultiSimilarityLoss',
            miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
            miner_margin=0.1,
            faiss_gpu=False
        )
        self.clipvpr_encoder.to(torch.device('cuda:0'))

        #load the model weights from the mode_p[ath]
        model_path = '/home/nikhil/Downloads/resnet50_epoch(78)_step(3555)_R1[0.9790]_R5[0.9925].ckpt'
        self.clipvpr_encoder.load_state_dict(torch.load(model_path)['state_dict'])
        self.clipvpr_encoder.to(torch.device('cuda:0')).eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        # FIXME: comment for debug
        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        # torch.cuda.empty_cache()
        # print("#" * 32)
        # self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
    
    def save_reconstruction(self, reconstruction_path):

        from pathlib import Path
        import random
        import string

        t = self.video.counter.value
        tstamps = self.video.tstamp[:t].cpu().numpy()
        images = self.video.images[:t].cpu().numpy()
        disps = self.video.disps_up[:t].cpu().numpy()
        poses = self.video.poses[:t].cpu().numpy()
        intrinsics = self.video.intrinsics[:t].cpu().numpy()

        Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
        np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
        np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
        np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
        np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
        np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)

