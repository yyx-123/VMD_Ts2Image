import os
import numpy as np
import pickle
import torch
from tqdm import tqdm
from TimeSeriesTransformer import TimeSeriesTransformer


Ts = 0.075
LEN = 400
t = np.linspace(0, Ts * LEN, LEN)


IMFDir = "../dataset/IMFs/"
datasetDir = "../dataset/images/"
imagesize = [16, 24, 32, 40, 48, 56, 64]
for sz in imagesize:
    dataset = []
    for SubId in tqdm(range(1, 31)):
        SubDir = str(SubId) + '/'
        for file in os.listdir(IMFDir + SubDir):
            # 读取IMF数据的任务信息
            fileName = file.split('.')[0]
            fileInfo = file.split('.')[0].split('_')
            taskNum = fileInfo[0].split('=')[1]
            task = fileInfo[1].split('=')[1]
            channel = fileInfo[2].split('=')[1]

            # 读取IMF
            originalIMF = np.load(IMFDir + SubDir + file)
            IMF = originalIMF.reshape(1, -1)

            # 有选择性的计算特征(GAF、MTF、tan、tanh、sigmoid、nonmapping)并融合
            featureSel = {'GAF': False,
                          'MTF': True,
                          'tan': True,
                          'tanh': False,
                          'sigmoid': False,
                          'nonMapping': False
                          }
            IMAGE_SIZE = sz
            BINs = 32

            transformer = TimeSeriesTransformer(ts=IMF, imageSize=IMAGE_SIZE)
            features = []
            datasetName = ""
            if featureSel['GAF']:
                GASF, GADF = transformer.GAF()
                features.append(GASF)
                features.append(GADF)
                datasetName += "GAF_"
            if featureSel['MTF']:
                MTF = transformer.MTF(nBins=BINs)
                features.append(MTF)
                datasetName += "MTF_"
            if featureSel['tan']:
                tanMappingSum, tanMappingDiff = transformer.tanMapping()
                features.append(tanMappingSum)
                features.append(tanMappingDiff)
                datasetName += "tan_"
            if featureSel['tanh']:
                tanhMappingSum, tanhMappingDiff = transformer.tanMapping()
                features.append(tanhMappingSum)
                features.append(tanhMappingDiff)
                datasetName += "tanh_"
            if featureSel['sigmoid']:
                sigmoidMappingSum, sigmoidMappingDiff = transformer.sigmoidMapping()
                features.append(sigmoidMappingSum)
                features.append(sigmoidMappingDiff)
                datasetName += "sigmoid_"
            if featureSel['nonMapping']:
                nonMappingSum, nonMappingDiff = transformer.nonMapping()
                features.append(nonMappingSum)
                features.append(nonMappingDiff)
                datasetName += "nonMapping_"

            feature = np.stack(tuple(features), axis=0)
            feature = torch.from_numpy(feature)

            # 保存结果
            dataInfo = {'SubId':SubId, 'taskNum':taskNum, 'channel':channel}
            label = int(task)
            data = (feature, label, dataInfo)
            dataset.append(data)

    # 持久化数据
    datasetName += str(sz)
    print(datasetName)
    with open(datasetDir + datasetName + '.pickle', 'wb') as f:
        pickle.dump(dataset, f)

