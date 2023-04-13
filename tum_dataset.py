import argparse
import os
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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
                    timestamp = round(float(values[0]), 2)
                    pose = np.array([float(x) for x in values[1:]])
                    self.gt_poses[timestamp] = pose
        
        # Step 2: Match the RGB image file names with the ground truth timestamps
        # note: the ground truth timestamps are rounded to 2 decimal places because
        # otherwise they do not match
        self.rgb_filenames = []
        self.gt_poses_list = []
        for filename in sorted(os.listdir(self.rgb_folder)):
            if filename.endswith('.png'):
                timestamp = round(float(os.path.splitext(filename)[0]), 2)
                if timestamp in self.gt_poses:
                    self.rgb_filenames.append(os.path.join(self.rgb_folder, filename))
                    self.gt_poses_list.append(self.gt_poses[timestamp])        

        # Step 3: Set the positive and negative images for each image
        self.set_positives_and_negatives_radius()

    def __len__(self):
        return len(self.rgb_filenames)
    
    def __getitem__(self, idx):
        rgb_file = self.rgb_filenames[idx]

        # Load the RGB image
        rgb_image = Image.open(rgb_file).convert('RGB')
        
        # Apply the transform, if provided
        if self.transform:
            rgb_image = self.transform(rgb_image)

        positives = self.positive_images[rgb_file]
        negatives = self.negative_images[rgb_file]
        return rgb_image, positives, negatives

    def set_positives_and_negatives_knn(self, k: int = 10):
        """
        Sets positive and negative images for all images in the dataset.
        """
        self.positive_images = {}
        self.negative_images = {}
        for i in range(len(self)):
            gt_pose = self.gt_poses_list[i]

            # given a query image, find the images that are nearby in distance
            # so the label for the image is its xyz coordinates
            X = np.array(list(self.gt_poses.keys())).reshape(-1, 1)
            y = np.array(self.gt_poses_list)
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X, y)

            # potential positives
            _, indices = knn.kneighbors([gt_pose], n_neighbors=k, return_distance=False)
            positive_indices = indices[0]
            positive_images = [self.rgb_filenames[j] for j in positive_indices]

            # definite negatives
            negative_indices = [j for j in range(len(self)) if j not in positive_indices]
            negative_images = [self.rgb_filenames[j] for j in negative_indices]

            # set the positive and negative images
            self.positive_images[self.rgb_filenames[i]] = positive_images
            self.negative_images[self.rgb_filenames[i]] = negative_images

    def set_positives_and_negatives_radius(self, radius: float = 5.0):
        """
        Sets positive and negative images for all images in the dataset.
        """
        for i in range(len(self)):
            gt_pose = self.gt_poses_list[i]

            for j in range(len(self)):
                if i != j:
                    gt_pose2 = self.gt_poses_list[j]
                    distance = np.linalg.norm(gt_pose - gt_pose2)
                    if distance <= radius:
                        self.positive_images[self.rgb_filenames[i]].append(self.rgb_filenames[j])
                    else:
                        self.negative_images[self.rgb_filenames[i]].append(self.rgb_filenames[j])


if __name__ == '__main__':    
    # Visualize the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='rgbd_dataset_freiburg1_360')
    args = parser.parse_args()

    # Define the dataset
    dataset = TUMRGBDDataset(data_folder=args.data_folder)
    print(len(dataset))
    # Define the data loader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Iterate through the data loader
    for rgb_images, positives, negatives in data_loader:
        # Print the shapes of the tensors
        print("Printing shapes of tensors:")
        print('RGB images:', rgb_images.shape)
        print('Positives:', positives.shape)
        print('Negatives:', negatives.shape)

        # Make a grid from the RGB images
        out = make_grid(rgb_images)

        # Convert the grid to numpy
        out = out.numpy()

        # Convert the channels from BGR to RGB
        out = np.transpose(out, (1, 2, 0))

        # Plot the images
        plt.figure(figsize=(10, 10))
        plt.imshow(out)
        plt.show()
