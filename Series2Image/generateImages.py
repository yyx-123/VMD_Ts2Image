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

imageDir = "../dataset/images/"
imagesize = [32, 64]
for sz in imagesize:
    for SubId in tqdm(range(1, 31)):
        dataset = []
        SubDir = str(SubId) + '/'
        for file in os.listdir(IMFDir + SubDir):
            # 读取IMF数据的任务信息
            fileName = file.split('.')[0]
            fileInfo = file.split('.')[0].split('_')
            taskNum = fileInfo[0].split('=')[1]
            task = fileInfo[1].split('=')[1]
            channel = fileInfo[2].split('=')[1]
            # 读取IMF数据
            originalIMF = np.load(IMFDir + SubDir + file)
            IMF = originalIMF.reshape(1, -1)

            # 有选择性的计算特征(GAF、MTF、tan、tanh、sigmoid、nonmapping)并融合
            featureSel = {'GAF': False,
                          'MTF': False,
                          'tan': False,
                          'tanh': False,
                          'sigmoid': False,
                          'linear': True
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
            if featureSel['linear']:
                linearSum, linearDiff = transformer.linear()
                features.append(linearSum)
                features.append(linearDiff)
                datasetName += "linear_"

            feature = np.stack(tuple(features), axis=0)
            feature = torch.from_numpy(feature)

            # 把当前一个file(.npy)的结果保存到总的dataset中
            dataInfo = {'SubId':SubId, 'taskNum':taskNum, 'channel':channel}
            label = int(task)
            data = (feature, label, dataInfo)
            dataset.append(data)

        # 持久化数据，这里是对每一个Sub的数据分别创立一个数据集，如果需要对全部被试创建数据集则需要将以下内容移至更外一层
        datasetName += str(sz)
        print(datasetName)

        tgtDir = imageDir + 'SubImages/' + SubDir
        if not os.path.exists(tgtDir):
            os.makedirs(tgtDir)

        with open(tgtDir + datasetName + '.pickle', 'wb') as f:
            pickle.dump(dataset, f)

