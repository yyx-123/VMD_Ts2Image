from TsDataset import TsDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import utils
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from models import LSTM_cls
from sklearn.metrics import confusion_matrix


def train(SubId=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using " + device.type)

    # 加载、划分数据集
    print("loading IMF dataset of SubId = {}....".format(SubId))
    dataset = TsDataset(datasetPath='../dataset/IMFs/', SubId=SubId)
    ts_len = dataset[0][0].shape[0]
    print(ts_len)

    TRAIN_PERCENT = 0.8
    train_size = int(TRAIN_PERCENT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    BATCH_SIZE = 64
    train_dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 加载模型
    # print("loading model ...")
    model = LSTM_cls(input_size=1, hidden_size=5, cls_num=3, ts_len=ts_len, batch_first=True).to(device)

    # 训练
    # print("start training ...")
    EPOCH_NUM = 151
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lossFunc = F.nll_loss
    msg = ""
    bestACC = -1
    for epoch in range(EPOCH_NUM):
        model.train()
        for train_data, labels in train_dataLoader:
            train_data = train_data.to(device)
            labels = labels.to(device)

            out = model(train_data)
            loss_train = lossFunc(out, labels)
            # Backpropagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()
        confusion = np.zeros([3, 3])
        for test_data, test_label in test_dataLoader:
            test_data = test_data.to(device)
            test_label = test_label.to(device)

            out = model(test_data)
            pred = torch.max(out, dim=1).indices
            for idx in range(len(test_label)):
                confusion[test_label[idx]][pred[idx]] += 1


        # 计算各项分类指标
        # 准确率：混淆矩阵对角线元素和（分类正确的样本数） / 总样本数
        acc = (confusion[0][0] + confusion[1][1] + confusion[2][2]) / len(test_dataLoader.dataset)
        prec0, prec1, prec2 = utils.calPrecision(confusion)
        recall0, recall1, recall2 = utils.calRecall(confusion)
        if acc > bestACC:
            msg = 'Epoch:{:2} | loss:{:.2f} | acc:{:.4f} | prec0:{:.4f} | prec1:{:.4f} | prec2:{:.4f} | recall0:{:.4f}  | recall1:{:.4f} | recall2:{:.4f}\n' \
              .format(epoch, loss_train, acc, prec0, prec1, prec2, recall0, recall1, recall2)
            # bestACC = acc
            print(msg)

    # 记录结果
    # resultDIr = "../results/"
    # with open(resultDIr + datasetName + ".txt", 'w') as result:
    #     result.write(msg)
    print(msg)
    return msg

if __name__ == '__main__':
    train(2)