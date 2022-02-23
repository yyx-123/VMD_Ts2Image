import scipy.io as scio
import numpy as np
import utils
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
from VMD import utils
from Series2Image.TimeSeriesTransformer import TimeSeriesTransformer



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
    plt.title('简单预处理后的fNIRS数据')

    plt.subplot(3,1,2)
    plt.plot(t, IMF)
    plt.title('由自适应VMD算法提取的神经活动信号分量，频率约为{:4f}Hz'.format(np.min(omega)))

    plt.subplot(3, 1, 3)
    plt.plot(t, noise)
    plt.title('心率噪声分量，频率约为{:4f}Hz'.format(np.max(omega)))
    plt.xlabel('(c)')

    plt.tight_layout()
    plt.savefig('图5 自适应VMD分解结果.pdf', format='pdf')
    plt.show()


def method2(SubId, taskNum, channel):
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
    print(np.min(omega))
    reshapedIMF = IMF.reshape(1, -1)

    transformer = TimeSeriesTransformer(ts=reshapedIMF, imageSize=96)
    GASF, GADF = transformer.GAF()
    MTF = transformer.MTF()
    linearSum, linearDiff = transformer.linear()
    sigmoidMappingSum, sigmoidMappingDiff = transformer.sigmoidMapping()
    tanMappingSum, tanMappingDiff = transformer.tanMapping()
    tanhMappingSum, tanhMappingDiff = transformer.tanMapping()


    plt.figure(figsize=(5,12))
    grid = plt.GridSpec(4,2,hspace=0.4, wspace=0.2)

    plt.subplot(grid[0, 0:2])
    plt.plot(np.linspace(0, 30, 400), IMF)
    plt.xlabel("t (s)")
    plt.title('preprocessed fNIRS signal')

    plt.subplot(grid[1, 0])
    plt.imshow(GASF)
    plt.axis('off')
    plt.title('GAF')
    plt.subplot(grid[1, 1])
    plt.imshow(MTF)
    plt.axis('off')
    plt.title('MTF')

    plt.subplot(grid[2, 0])
    plt.imshow(linearSum)
    plt.axis('off')
    plt.title('Linear Mapping Field(LMF)')
    plt.subplot(grid[2, 1])
    plt.imshow(sigmoidMappingSum)
    plt.axis('off')
    plt.title('Sigmoid Mapping Field')

    plt.subplot(grid[3, 0])
    plt.imshow(tanMappingSum)
    plt.axis('off')
    plt.title('Tan Mapping Field')
    plt.subplot(grid[3, 1])
    plt.imshow(tanhMappingSum)
    plt.axis('off')
    plt.title('Tanh Mapping Field')

    plt.show()



if __name__ == '__main__':
    #method1(SubId=9, taskNum=3, channel=1)
    method2(SubId=9, taskNum=15, channel=6)
    method2(SubId=9, taskNum=3, channel=1)