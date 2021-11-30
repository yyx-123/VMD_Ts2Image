import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from HRF import task_HRF
from pyts.approximation import PiecewiseAggregateApproximation

def series2GAF(ts, imageSize, scale='[0,1]'):
    n = len(ts)

    # step 1: 标准化时间序列
    min_ts, max_ts = np.min(ts), np.max(ts)
    if scale == '[0,1]':
        diff = max_ts - min_ts
        if diff != 0:
            rescaled_ts = (ts - min_ts) / diff
    if scale == '[-1,1]':
        diff = max_ts - min_ts
        if diff != 0:
            rescaled_ts = (2 * ts - diff) / diff

    # step 2: PAA压缩长度
    paa = PiecewiseAggregateApproximation(window_size=None, output_size=imageSize, overlapping=False)
    rescaled_ts = paa.fit_transform(rescaled_ts)

    # step 3: 计算GAF
    sin_ts = np.sqrt(np.clip(1 - rescaled_ts ** 2, 0, 1))
    # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
    GASF = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
    # sin(x1-x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
    GADF = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)

    # step4: 输出结果
    return GASF, GADF

def visualizeGAFResult(ts, GASF, GADF, title=None):
    Ts = 0.075
    LEN = 400
    t = np.linspace(0, Ts * LEN, LEN)

    HRF_GASF, HRF_GADF = series2GAF(task_HRF().reshape(1, -1), imageSize=np.shape(GASF)[0])
    plt.subplot(2, 3, 1)
    plt.plot(t, task_HRF())
    plt.subplot(2, 3, 2)
    plt.imshow(HRF_GASF, cmap="rainbow")
    plt.subplot(2, 3, 3)
    plt.imshow(HRF_GADF, cmap="rainbow")

    plt.subplot(2, 3, 4)
    plt.plot(t, ts)
    plt.subplot(2, 3, 5)
    plt.imshow(GASF, cmap="rainbow")
    plt.subplot(2, 3, 6)
    plt.imshow(GADF, cmap="rainbow")
    if title != None:
        plt.suptitle(title)
    plt.show()


# test
if __name__ == '__main__':
    HRF = task_HRF().reshape(1, -1)
    ts1 = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5]).reshape(1, -1)
    ts2 = np.array([5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]).reshape(1, -1)

    GASF, GADF = series2GAF(HRF, imageSize=64)
    visualizeGAFResult(task_HRF(), GASF, GADF)
