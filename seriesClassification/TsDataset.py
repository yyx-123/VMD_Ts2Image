import os
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
import numpy as np
import torch
from torch.utils.data import Dataset

class TsDataset(Dataset):
    def __init__(self, datasetPath, SubId):
        self.IMFdir = datasetPath + str(SubId) + '/'

        self.dataset = self.loadData(self.IMFdir)

    def __getitem__(self, index):
        ts = self.dataset[index][0]
        label = self.dataset[index][1]

        return ts, label

    def __len__(self):
        return len(self.dataset)

    def loadData(self, dir):
        dataset = []
        for file in os.listdir(dir):
            ts = np.load(dir + file)
            # 排除掉过短的时间序列后取前397个数据，再降采样，可以确保每一个数据的长度均为133
            if len(ts) < 397:
                continue
            else:
                ts = ts[:397][::3]      # [::3]相当于ts[0:ts.size:3])，降采样为原来的三分之一
                ts = ts.reshape(-1, 1)
            ts = torch.from_numpy(ts).float()   # 转换成tensor并将数据类型转为float32
            label = file.split('.')[0].split('_')[1].split('=')[1]

            data = (ts, int(label) - 1)
            dataset.append(data)

        return dataset

if __name__ == '__main__':
    dataset = TsDataset("../dataset/IMFs/", SubId=1)