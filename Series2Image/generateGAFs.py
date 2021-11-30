import os
import numpy as np
import pickle
from tqdm import tqdm
from myGAF import series2GAF, visualizeGAFResult



Ts = 0.075
LEN = 400
t = np.linspace(0, Ts * LEN, LEN)

IMFDir = "../VMD/IMFs/"
GAFDir = "GAF/"
for SubId in tqdm(range(1, 31)):
    SubDir = str(SubId) + '/'
    for file in tqdm(os.listdir(IMFDir + SubDir)):
        # 读取IMF数据的任务信息
        fileName = file.split('.')[0]
        fileInfo = file.split('.')[0].split('_')
        taskNum = fileInfo[0].split('=')[1]
        task = fileInfo[1].split('=')[1]
        channel = fileInfo[2].split('=')[1]

        # 读取IMF
        originalIMF = np.load(IMFDir + SubDir + file)
        IMF = originalIMF.reshape(1, -1)

        # 计算GAF
        IMAGE_SIZE = 64
        GASF, GADF = series2GAF(ts=IMF, imageSize=IMAGE_SIZE, scale='[0,1]')
        # 可视化结果
        # visualizeGAFResult(originalIMF, GASF, GADF, title=fileName)

        # 保存结果
        data = {'series': originalIMF, 'GASF': GASF, 'GADF': GADF}
        if not os.path.exists(GAFDir + SubDir):
            os.mkdir(GAFDir + SubDir)
        with open(GAFDir + SubDir + fileName + '.pickle', 'wb') as f:
            pickle.dump(data, f)

