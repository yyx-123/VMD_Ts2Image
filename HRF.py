import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import convolve as conv
from math import exp

Ts = 0.075

# 计算并绘制单次刺激（第0秒）下的典型HRF
a1 = 6
a2 = 16
t_HRF = np.linspace(start=0, stop=30, num=int(30 / Ts))
HRF = np.zeros(len(t_HRF))
for i in range(len(t_HRF)):
    T = t_HRF[i]
    HRF[i] = (T**(a1-1) * exp(-T)) / gamma(a1) - (T**(a2-1) * exp(-T)) / (6 * gamma(a2))
plt.subplot(3,1,1)
plt.plot(t_HRF, HRF)

# 计算并绘制任务参考波
REF = np.zeros(int(60 / Ts))
for i in range(0, int(10 / Ts)):
    REF[i] = 1
for i in range(int(30 / Ts), int(40 / Ts)):
    REF[i] = 1
plt.subplot(3,1,2)
plt.plot(np.linspace(0, 60, int(60 / Ts)), REF)

# 计算并绘制任务下的典型HRF，通过单个刺激下的HRF与任务参考波卷积得到
HRF_refed = conv(HRF, REF)[ : int(70 / Ts)]
t = np.linspace(start=0, stop=70, num=int(70 / Ts))
plt.subplot(3,1,3)
plt.plot(t, HRF_refed)


plt.show()
