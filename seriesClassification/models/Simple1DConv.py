import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DConv(nn.Module):
    def __init__(self, clsNum):
        super().__init__()
        self.name = 'Simple1DConv'

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.FC1 = nn.Linear(in_features=70, out_features=30)
        self.FC2 = nn.Linear(in_features=30, out_features=clsNum)


    def forward(self, x):
        x = x.type(torch.FloatTensor).to(torch.device('cuda'))

        batchSize = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(batchSize, 1, -1)

        x = self.FC1(x)
        x = self.FC2(x)

        x = x[:, 0, :]
        return F.log_softmax(x, dim=1)


