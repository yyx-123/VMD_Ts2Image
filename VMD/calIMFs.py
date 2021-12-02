import scipy.io as scio
import numpy as np
import utils
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 提取任务数据
taskdata = scio.loadmat("../dataset/rawData/taskInfo.mat")
triggers = taskdata['triggers']         # 每一行是一个被试的任务开始采样点编号，30*75
event = taskdata['event'][0]            # 1 2 3 3 2 1 ... 的序列，描述了任务性质
event_desc = taskdata['event_desc'][0]  # 对1、2、3的描述
LEN = 400
Ts = 0.075
fs = 1 / Ts
t = np.linspace(0, Ts*LEN, LEN)

# 提取每一个被试的近红外数据
for SubId in range(1, 31):
    print(SubId)
    SubDir = str(SubId) + "\\"
    if SubId <= 9:
        nirsdata = scio.loadmat('..\\dataset\\Sub0' + str(SubId) + '.mat')['nirsdata'][0][0][0]
    else:
        nirsdata = scio.loadmat('..\\dataset\\Sub' + str(SubId) + '.mat')['nirsdata'][0][0][0]

    # 对于每一次任务，提取信号起始点
    for taskNum in tqdm(range(75)):
        startpoint = triggers[SubId - 1][taskNum]

        # 对每一通道的数据进行划分、计算
        for channel in range(20):
            data = nirsdata[:, channel][startpoint: startpoint + LEN]
            ALPHA = 150

            K = utils.findBestK(data=data, fs=fs, ALPHA=ALPHA, Kmin=2, Kmax=8)
            if K > 0:
                u, omega, corr, IO = utils.myVMD(data=data, fs=fs, ALPHA=ALPHA, K=K)
                u = u[np.argmin(omega), :]

                # 保存结果
                tgtDir = "..\VMD\IMFs\\"
                if not os.path.exists(tgtDir + SubDir):
                    os.mkdir(tgtDir + SubDir)
                fileName = "taskNum={}_task={}_channel={}".format(taskNum, event[taskNum], channel)
                np.save(tgtDir + SubDir + fileName, u)
                
