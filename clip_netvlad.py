import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, Resize, ToTensor

from netvlad import NetVLAD
from netvlad import EmbedNet
from hard_triplet_loss import HardTripletLoss

import clip
from PIL import Image

from tum_dataset import TUMRGBDDataset

# clip-netvlad as a torch nn module
class CLIPNetVLAD(nn.Module):
    def __init__(self, num_clusters: int, alpha: float, device="cuda"):
        super().__init__()
        self.device = device
        self.clip, self.preprocess = clip.load("ViT-B/32", device=device)
        dim = list(self.clip.visual.parameters())[-1].shape[0]  # last channels (512)
        self.net_vlad = NetVLAD(num_clusters=num_clusters, dim=dim, alpha=alpha)

    def forward(self, x):
        # x is a list of images
        # each image is a 3D tensor
        # first dimension is batch size
        # second dimension is number of channels
        # third dimension is number of pixels
        # x = torch.cat(x, 0)
        # x = self.preprocess(x)
        # x = self.clip.encode_image(x)
        # x = self.net_vlad(x)
        # return x
        with torch.no_grad():
            x = self.preprocess(x).unsqueeze(0).to(self.device)
            x = self.clip.encode_image(x)
            x /= x.norm(dim=-1, keepdim=True)
        return self.net_vlad(x)


def run_clip_netvlad_example(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    data_transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = TUMRGBDDataset(args.training_dir, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # model
    clip_netvlad = CLIPNetVLAD(num_clusters=32, alpha=1.0, device=device)
    optimizer = torch.optim.Adam(clip_netvlad.parameters(), lr=1e-3)

    # loss
    criterion = HardTripletLoss(margin=0.1).cuda()

    # train
    for epoch in range(args.num_epochs):
        for batch_idx, (images, poses) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_poses = clip_netvlad(images)
            triplet_loss = criterion(predicted_poses, poses)
            # triplet_loss = criterion(clip_netvlad(img), labels)
            triplet_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {triplet_loss.item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dir', type=str, required=True, help="path to training imgs")
    parser.add_argument('--gt-file', type=str, required=True, help="path to ground truth file")
    parser.add_argument('--num-epochs', type=int, default=10, help="number of epochs to train for")
    args = parser.parse_args()
    
    run_clip_netvlad_example(args)
