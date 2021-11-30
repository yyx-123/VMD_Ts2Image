import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def series2GAF(ts, scale='[0,1]'):
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

    # step 2: 计算GAF
    sin_ts = np.sqrt(np.clip(1 - rescaled_ts ** 2, 0, 1))
    # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
    GASF = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
    # sin(x1-x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
    GADF = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)

    # step3: 输出结果
    return GASF, GADF


# test
if __name__ == '__main__':
    ts1 = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5]).reshape(1, -1)
    ts2 = np.array([5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]).reshape(1, -1)

    ts1_GASF, ts1_GADF = series2GAF(ts1)
    plt.imshow(ts1_GADF)
    plt.show()
