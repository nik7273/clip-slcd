import argparse
import os
import h5py    
import numpy as np
import torch    
from torch.utils.data import Dataset, DataLoader

class HLocDataset(Dataset):
    def __init__(self, data_folder: str, transform=None):
        self.data_folder = data_folder
        self.length = len(os.listdir(self.data_folder))
        for filename in os.listdir(self.data_folder):
            if not filename.endswith('.hdf5'):
                print("Please make sure there are only .hdf5 files in your data folder")
                #f1 = h5py.File(filename,'r+')

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int):
        # get idxth hdf5 file
        place_fname = os.listdir(self.data_folder)[idx] # assumption is that folder ONLY contains relevant .hdf5 files
        place_file = h5py.File(os.path.join(self.data_folder, place_fname), 'r+')
        place = torch.tensor(np.array(place_file['rgb']))
        label = idx # wtf???
        return place, label
    

def example(args: argparse.Namespace):
    """
    Example of using the HLocDataset class in a training loop
    """
    dataset = HLocDataset(args.data_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader):
        places, labels = data
        print(places.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='data/1pXnuDYAj8r/1pXnuDYAj8r_point0.hdf5')
    args = parser.parse_args()
    example(args)
