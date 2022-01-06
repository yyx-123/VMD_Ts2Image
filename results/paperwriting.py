import scipy.io as scio
import numpy as np
import utils
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from VMD import utils



# 该方法提取Sub在完成taskNum任务时第channel通道的原始fNIRS数据、滤波后的fNIRS数据和提取出的IMF分量并进行可视化
def method1(SubId, taskNum, channel):
    # 提取原始数据
    if SubId <= 9:
        raw = scio.loadmat('D:/fnirs dataset/tapping dataset/0raw/fNIRS0' + str(SubId) + '.mat')
    else:
        raw = scio.loadmat('D:/fnirs dataset/tapping dataset/0raw/fNIRS' + str(SubId) + '.mat')
    raw = raw['ch' + str(channel)][26 + 400 * (taskNum - 1) : 26 + 400 * taskNum]
    # 提取滤波后的数据
    if SubId <= 9:
        preprocessed = scio.loadmat('../dataset/rawData/Sub0' + str(SubId) + '.mat')['nirsdata'][0][0][0]
    else:
        preprocessed = scio.loadmat('../dataset/rawData/Sub' + str(SubId) + '.mat')['nirsdata'][0][0][0]
    preprocessed = preprocessed[26 + 400 * (taskNum - 1) : 26 + 400 * taskNum, channel - 1]
    # 计算IMF
    ALPHA = 150
    K = utils.findBestK(data=preprocessed, fs=13.3333, ALPHA=ALPHA, Kmin=2, Kmax=8)
    u, omega, corr, IO = utils.myVMD(data=preprocessed, fs=13.3333, ALPHA=ALPHA, K=K)
    IMF = u[np.argmin(omega), :]
    noise = u[np.argmax(omega), :]


    # 开始作图
    t = np.linspace(0, 30, 400)

    plt.subplot(3,1,1)
    plt.plot(t, preprocessed)
    plt.title('preprocessed fNIRS data')
    plt.subplot(3,1,2)
    plt.plot(t, IMF)
    plt.title('interested IMF extracted by self-adaptive VMD, {:4f}Hz'.format(np.min(omega)))
    plt.subplot(3, 1, 3)
    plt.plot(t, noise)
    plt.title('noise IMF, {:4f}Hz'.format(np.max(omega)))

    plt.tight_layout()
    plt.show()


    print(123)





if __name__ == '__main__':
    method1(SubId=9, taskNum=3, channel=1)