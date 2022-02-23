import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class LSTM_cls(nn.Module):
    def __init__(self, input_size, hidden_size, cls_num, ts_len, batch_first=True):
        super(LSTM_cls, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=batch_first
        )

        self.FC = nn.Sequential(
            nn.Linear(hidden_size * ts_len, 256),
            nn.Linear(256, 32),
            nn.Linear(32, cls_num)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(lstm_out.shape[0], 1, -1)
        embedding = self.FC(lstm_out)
        embedding = torch.squeeze(embedding, dim=1)
        logits = F.log_softmax(embedding, dim=1)

        return logits

class RNN_cls(nn.Module):
    def __init__(self, input_size, hidden_size, cls_num, ts_len, batch_first=True):
        super(RNN_cls, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=batch_first
        )

        self.FC = nn.Sequential(
            nn.Linear(hidden_size * ts_len, 256),
            nn.Linear(256, 32),
            nn.Linear(32, cls_num)
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out.reshape(rnn_out.shape[0], 1, -1)
        embedding = self.FC(rnn_out)
        embedding = torch.squeeze(embedding, dim=1)
        logits = F.log_softmax(embedding, dim=1)

        return logits

class GRU_cls(nn.Module):
    def __init__(self, input_size, hidden_size, cls_num, ts_len, batch_first=True):
        super(GRU_cls, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=batch_first
        )

        self.FC = nn.Sequential(
            nn.Linear(hidden_size * ts_len, 256),
            nn.Linear(256, 32),
            nn.Linear(32, cls_num)
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out.reshape(rnn_out.shape[0], 1, -1)
        embedding = self.FC(rnn_out)
        embedding = torch.squeeze(embedding, dim=1)
        logits = F.log_softmax(embedding, dim=1)

        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # batchsize = 100, 序列长度400， 维度为1
    x = torch.randn(64, 133, 1, device='cuda')

    # model = nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True).to(device)
    # embedding , _ = model(x)
    # print(embedding.shape)

    model = RNN_cls(input_size=1, hidden_size=5, cls_num=3).to(device)
    logits = model(x)
    print(logits.shape)