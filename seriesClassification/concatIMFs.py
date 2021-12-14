import os
import torch
import pickle
import numpy as np
from tqdm import tqdm

IMFDir = '../dataset/IMFs/'
dataset = []
for SubId in tqdm(range(1, 31)):
    SubDir = IMFDir + str(SubId) + '/'
    for IMFFileName in os.listdir(SubDir):
        IMFdata = torch.from_numpy(np.load(SubDir + IMFFileName))
        taskLabel = IMFFileName.split('_')[1].split('=')[1]

        data = (IMFdata, taskLabel)
        dataset.append(data)

tgtDir = '../dataset/IMFs/'
with open(tgtDir + 'IMFs' + '.pickle', 'wb') as f:
    pickle.dump(dataset, f)