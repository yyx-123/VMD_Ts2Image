import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = self.loadData()

    def __getitem__(self, index):
        data = self.dataset[index][0]
        task = self.dataset[index][1]

        label = task - 1
        # if task == 1:
        #     label = np.array([1, 0, 0])
        # elif task == 2:
        #     label = np.array([0, 1, 0])
        # else:
        #     label = np.array([0, 0, 1])
        # label = torch.from_numpy(label)

        return data, label

    def __len__(self):
        return len(self.dataset)

    def loadData(self):
        with open(self.path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset


