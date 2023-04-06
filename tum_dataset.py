import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class TUMRGBDDataset(Dataset):
    def __init__(self, data_folder: str, transform=None):
        self.transform = transform
        self.rgb_folder = os.path.join(data_folder, 'rgb')
        self.gt_file = os.path.join(data_folder, 'groundtruth.txt')
        
        # Step 1: Parse the ground truth file
        self.gt_poses = {}
        with open(self.gt_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    values = line.split()
                    timestamp = str(float(values[0]))
                    pose = np.array([float(x) for x in values[1:]])
                    self.gt_poses[timestamp] = pose
        
        # Step 2: Match the RGB image file names with the ground truth timestamps
        self.rgb_filenames = []
        self.gt_poses_list = []
        for filename in sorted(os.listdir(self.rgb_folder)):
            if filename.endswith('.png'):
                timestamp = float(os.path.splitext(filename)[0])
                if timestamp in self.gt_poses:
                    self.rgb_filenames.append(os.path.join(self.rgb_folder, filename))
                    self.gt_poses_list.append(self.gt_poses[timestamp])
    

        import re

        def find_closest_key(dict_obj, key_str):
            # Filter keys that match a decimal number pattern
            regex = re.compile(r'^\d+\.\d+$')
            filtered_keys = [k for k in dict_obj.keys() if regex.match(k)]

            # Find the closest key
            closest_key = min(filtered_keys, key=lambda k: abs(float(k) - float(key_str)))
            return closest_key

        import pdb; pdb.set_trace()
    def __len__(self):
        return len(self.rgb_filenames)
    
    def __getitem__(self, idx):
        rgb_file = self.rgb_filenames[idx]
        gt_pose = self.gt_poses_list[idx]

        # Load the RGB image
        rgb_image = Image.open(rgb_file).convert('RGB')
        
        # Apply the transform, if provided
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # Convert the pose to a PyTorch tensor
        gt_pose = torch.from_numpy(gt_pose).float()
        
        return rgb_image, gt_pose

