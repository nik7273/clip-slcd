import argparse
import os
import h5py    
import numpy as np
import torch    
import glob 
from torch.utils.data import Dataset, DataLoader

class HLocDataset(Dataset):
    def __init__(self, data_folder: str, transform=None, imgs_per_place=10):
        self.data_folder = data_folder
        self.filenames = list(glob.glob(os.path.join(self.data_folder, '*','*.hdf5'), recursive=True))
        self.length = len(self.filenames)
        self.imgs_per_place = imgs_per_place
        for filename in self.filenames:
            if not filename.endswith('.hdf5'):
                print("Please make sure there are only .hdf5 files in your data folder")
                #f1 = h5py.File(filename,'r+')

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int):
        # get idxth hdf5 file
        place_fname = self.filenames[idx] # assumption is that folder ONLY contains relevant .hdf5 files
        place_file = h5py.File(place_fname, 'r+')
        place = torch.tensor(np.array(place_file['rgb'])).permute(0, 3, 1, 2)
        if self.imgs_per_place < place.shape[0]:
            #randomly choose self.imgs_per_place images 
            choices = np.random.choice(np.arange(place.shape[0]), self.imgs_per_place, replace=False)
            place = place[choices] 
        else:
            choices = np.random.choice(np.arange(place.shape[0]), self.imgs_per_place, replace=True)
            place = place[choices]
        label = idx # wtf???
        return place, label
    

def example(args: argparse.Namespace):
    """
    Example of using the HLocDataset class in a training loop
    """
    dataset = HLocDataset(args.data_folder)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader):
        places, labels = data
        print(places.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='/mnt/syn/advaiths/datasets/HPointLoc')
    args = parser.parse_args()
    example(args)
