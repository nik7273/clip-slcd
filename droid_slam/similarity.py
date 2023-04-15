import numpy as np
import glob
import os

def add_edges(datapath):
    # edge_list = [['1305031910.765238','1305031911.035050'],
    #              ['1305031911.633337','1305031911.097196'],
    #              ['1305031912.096953','1305031912.965034'],
    #              ['1305031912.965034','1305031911.035050']]
    edge_list = [['1305031910.765238','1305031911.035050'],['1305031911.633337','1305031911.097196']]
    
    return timestamp2idx(datapath, edge_list)

def timestamp2idx(datapath, edge_list):
    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::2]
    tstamps = np.array([x.split('/')[-1][:-4] for x in images_list])
    edge_list = np.array(edge_list)
    edge_list_idx = []
    for i, edge in enumerate(edge_list):
        idx1 = np.where(tstamps==edge[0])
        idx2 = np.where(tstamps==edge[1])
        
        # # follow the same convention
        # if (idx1[0][0]%2!=0) & (idx2[0][0]%2!=0):
        idx1 = int(idx1[0][0])
        idx2 = int(idx2[0][0])
        edge_list_idx.append([max(idx1, idx2), min(idx1, idx2)])
        edge_list_idx.append([min(idx1, idx2), max(idx1, idx2)])
        
    return edge_list_idx
    
    

if __name__ == '__main__':
    # datapath = "datasets/TUM-RGBD/rgbd_dataset_freiburg1_room_copy"
    # extra_edges = add_edges(datapath)
    # print(extra_edges)
    a = np.array([[1, 2, 3], [2, 3, 4]])
    idx = np.array([0, 1])
    print(a[idx])