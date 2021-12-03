import os
import numpy as np
import pickle
import torch
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler
from myGAF import series2GAF, visualizeGAFResult
from pyts.image import MarkovTransitionField



Ts = 0.075
LEN = 400
t = np.linspace(0, Ts * LEN, LEN)


IMFDir = "../VMD/IMFs/"
datasetDir = "../dataset/"
imagesize = [16, 24, 32, 40, 48, 56]
for sz in imagesize:
    print(sz)
    datasetName = "GAF_MTF_" + str(sz) + "_dataset"
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

            # 计算特征(GAF和MTF)并融合
            IMAGE_SIZE = 48
            BINs = 32

            GASF, GADF = series2GAF(ts=IMF, imageSize=IMAGE_SIZE, scale='[0,1]')
            mtfModel = MarkovTransitionField(image_size=IMAGE_SIZE, n_bins=BINs)
            MTF = mtfModel.fit_transform(IMF)[0]

            feature = np.stack((GASF, GADF, MTF), axis=0)
            feature = torch.from_numpy(feature)

            # 可视化结果
            # visualizeGAFResult(originalIMF, GASF, GADF, title=fileName)

            # 保存结果
            dataInfo = {'SubId':SubId, 'taskNum':taskNum, 'channel':channel}
            label = int(task)
            data = (feature, label, dataInfo)

            dataset.append(data)
            # data = {'GASF': GASF, 'GADF': GADF, 'MTF': MTF}
            # if not os.path.exists(GAFDir + SubDir):
            #     os.mkdir(GAFDir + SubDir)
            # with open(GAFDir + SubDir + fileName + '.pickle', 'wb') as f:
            #     pickle.dump(data, f)

    # 持久化数据
    with open(datasetDir + datasetName + '.pickle', 'wb') as f:
        pickle.dump(dataset, f)

