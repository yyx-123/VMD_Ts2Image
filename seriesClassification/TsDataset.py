import pickle
from torch.utils.data import Dataset
from pyts.approximation import PiecewiseAggregateApproximation


class TsDataset(Dataset):
    def __init__(self, path='../dataset/IMFs/IMFs.pickle'):
        self.path = path
        self.dataset = self.loadData()

    def __getitem__(self, index):
        paa = PiecewiseAggregateApproximation(window_size=None, output_size=128, overlapping=False)

        data = self.dataset[index][0].reshape(1, -1)
        data = paa.fit_transform(data)
        task = self.dataset[index][1]

        label = int(task) - 1

        return data, label

    def __len__(self):
        return len(self.dataset)

    def loadData(self):
        with open(self.path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

