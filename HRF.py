import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import convolve as conv
from math import exp
from pyts.image import GramianAngularField

Ts = 0.075

# 计算并绘制单次刺激（第0秒）下的典型HRF
def single_HRF():
    a1 = 6
    a2 = 16
    t_HRF = np.linspace(start=0, stop=30, num=int(30 / Ts))
    HRF = np.zeros(len(t_HRF))
    for i in range(len(t_HRF)):
        T = t_HRF[i]
        HRF[i] = (T**(a1-1) * exp(-T)) / gamma(a1) - (T**(a2-1) * exp(-T)) / (6 * gamma(a2))
    return HRF

# 计算并绘制任务参考波
def task_REF(taskDuration=10, restDuration=20, taskEndSecond=30):
    REF = np.zeros(int(taskEndSecond / Ts))
    for i in range(0, int(taskDuration / Ts)):
        REF[i] = 1
    return REF

# 计算并绘制任务下的典型HRF，通过单个刺激下的HRF与任务参考波卷积得到
def task_HRF(taskDuration=10, restDuration=20, taskEndSecond=30):
    HRF_refed = conv(single_HRF(), task_REF())[ : int(taskEndSecond / Ts)]
    return HRF_refed

def plotHRFs():
    plt.subplot(3, 1, 1)
    t_HRF = np.linspace(start=0, stop=30, num=int(30 / Ts))
    plt.plot(t_HRF, single_HRF())

    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0, 30, int(30 / Ts)), task_REF())

    t = np.linspace(start=0, stop=30, num=int(30 / Ts))
    plt.subplot(3,1,3)
    plt.plot(t, task_HRF())
    plt.show()

def GAF4taskHRF(imgsz=64):
    data = task_HRF().reshape(1, -1)
    image_size = 64
    gasf = GramianAngularField(image_size=image_size, method='summation')
    data_gasf = gasf.fit_transform(data)
    gadf = GramianAngularField(image_size=image_size, method='difference')
    data_gadf = gadf.fit_transform(data)

    images = [data_gasf[0], data_gadf[0]]
    titles = ['Summation', 'Difference']
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    for image, title, ax in zip(images, titles, axs):
        ax.imshow(image, cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
        ax.set_title(title)
    plt.show()

GAF4taskHRF()
