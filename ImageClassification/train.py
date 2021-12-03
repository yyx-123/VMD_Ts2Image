import torch
from MyDataset import MyDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models.ResNet18_cls import ResNet18_cls


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using " + device.type)

# 加载、划分数据集
print("loading dataset....")
dataset = MyDataset(path='../dataset/GAF_MTF_64_dataset.pickle')
print(len(dataset))

TRAIN_PERCENT = 0.8
TEST_PERCENT = 1 - TRAIN_PERCENT
train_size = int(TRAIN_PERCENT * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

BATCH_SIZE = 64
train_dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 加载模型
model = ResNet18_cls(clsNum=3).to(device)

# 训练
EPOCH_NUM = 128
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lossFunc = F.nll_loss
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
    correct_cnt = 0
    confusion = np.zeros([3, 3])
    for test_data in test_dataLoader:
        test_data = test_data.to(device)
        out = model(test_data)
        pred = torch.max(out, dim=1).indices
        for idx in range(len(test_data['y'])):
            # 使用混淆矩阵计算
            confusion[test_data['y'][idx]][pred[idx]] += 1
            # 三分类
            # if pred[idx] == test_data['y'][idx]:
            #     correct_cnt += 1

    # 计算各项分类指标
    # 准确率：混淆矩阵对角线元素和（分类正确的样本数） / 总样本数
    acc = (confusion[0][0] + confusion[1][1] + confusion[2][2]) / len(test_dataLoader.dataset)

    # 输出结果
    print('Epoch:{} | loss:{:.2f} | acc:{:.4f}'.format(epoch, loss_train, acc))