import argparse
import os
import h5py    
import numpy as np
import torch    
import torchvision.transforms as T
import torchvision
#import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob 
from torch.utils.data import Dataset, DataLoader

HARDCODE_PATH = '/mnt/syn/advaiths/datasets/HPointLoc'

class HPointLocDataset(Dataset):
    def __init__(self, data_folder=HARDCODE_PATH, transform=None, llm_transform=None, imgs_per_place=10):
        self.data_folder = data_folder
        self.filenames = list(glob.glob(os.path.join(self.data_folder, '*','*.hdf5'), recursive=True))
        self.length = len(self.filenames)
        self.imgs_per_place = imgs_per_place
        self.transform = transform
        self.llm_transform = llm_transform
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
        place = np.array(place_file['rgb'])
        if self.imgs_per_place < place.shape[0]:
            #randomly choose self.imgs_per_place images 
            choices = np.random.choice(np.arange(place.shape[0]), self.imgs_per_place, replace=False)
            place = place[choices] 
        else:
            choices = np.random.choice(np.arange(place.shape[0]), self.imgs_per_place, replace=True)
            place = place[choices]
        label = torch.tensor([idx]).repeat(self.imgs_per_place) # wtf???
        llm_place = []
        place_list = []
        for p in place:
            tmp = T.ToPILImage()(p).convert('RGB')
            tmp2 = self.transform(tmp)
            place_list.append(tmp2.unsqueeze(0))
            llm_place.append(self.llm_transform(tmp).unsqueeze(0))
        llm_place = torch.cat(llm_place, dim=0)
        place = torch.cat(place_list, dim=0)
        return place, llm_place, label
    

def example(args: argparse.Namespace):
    """
    Example of using the HLocDataset class in a training loop
    """
    dataset = HLocDataset(args.data_folder)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=6)
    for i, data in enumerate(dataloader):
        places, labels = data
        #create  a 5x2 grid of images in places and convert them all to numpy 
        grid = torchvision.utils.make_grid(places[0], nrow=5, padding=2)
        grid = grid.permute(1, 2, 0).numpy()
        plt.imshow(grid)
        plt.savefig("hi.png")
        print(places.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='/mnt/syn/advaiths/datasets/HPointLoc')
    args = parser.parse_args()
    example(args)
