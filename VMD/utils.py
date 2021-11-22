import scipy.io as scio
from scipy.stats import pearsonr
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD

def findBestK(data, fs, ALPHA, Kmin=2, Kmax=10,TAU=0.00001, DC=0, INIT=0, TOL=1e-7):
    '''
    读取可选的K的范围，并从中挑选出最优的K值
    :param data: 待分解的数据
    :param fs: 信号的采样率
    :param ALPHA: 重要参数alpha
    :param Kmin: 重要参数K值的下限，默认为2
    :param Kmax: 重要参数K值的上限，默认为10
    :param TAU: 一般参数
    :param DC: 一般参数
    :param INIT: 一般参数
    :param TOL: 一般参数
    :return:
    '''

    # LEN = 400
    # Ts = 0.075
    # fs = 1 / Ts
    # t = np.linspace(0, Ts * LEN, LEN)
    # plt.plot(t,data)
    # plt.show()

    suitable_K = []
    suitable_IO = []
    suitable_omega = []
    proper_K = []
    proper_IO = []
    proper_omega = []
    for K in range(Kmin, Kmax):
        u, omega, corr, IOvalue = myVMD(data, fs, ALPHA, K)
        # plt.plot(t, u[np.argmin(omega),:])
        # plt.show()
        # 如果IO非常低，且频段正好落在适合的范围内则可以直接返回，不需要再多做了
        if IOvalue < 0.015:
            return K

        # imf的频段不属于fNIRS生理学上的频段；或者IO值太高（高于0.1即10%）划分正交性不够强。以上两种情况均不作为一次有效划分

        # 硬性条件没满足，直接pass这个K
        if IOvalue > 0.1 or not(0.01 < min(omega) < 0.07):
            continue
        # 这个范围里的数值虽然满足硬性条件但是也没多好，放到备选区proper
        elif (0.06 <= IOvalue <= 0.1) and (0.02 < min(omega) < 0.065):
            proper_K.append(K)
            proper_IO.append(IOvalue)
            proper_omega.append(min(omega))
        # 这个范围里的就还不错了，放在suitable区（0.2 < omega < 0.065 且 IO <0.06）
        elif IOvalue < 0.06 and (0.02 < min(omega) < 0.065):
            suitable_K.append(K)
            suitable_IO.append(IOvalue)
            suitable_omega.append(min(omega))

    # 在suitable区和proper区中找合适的K
    if len(suitable_K) == 0:
        if len(proper_K) == 0:
            return -1
        elif len(proper_K) == 1:
            return proper_K[0]
    # 只有一个合适的K，则返回这个K的划分
    elif len(suitable_K) == 1:
        return suitable_K[0]

    # 走到此处说明有不止一个符合条件的K需要判断
    if len(suitable_K) == 0 and len(proper_K) > 1:
        suitable_K = proper_K
        suitable_IO = proper_IO
        suitable_omega = proper_omega
    # 判断的原则是优先保证omega接近0.035
    feature = []
    A = 5
    for i in range(len(suitable_K)):
        feature.append( (A / abs(suitable_omega[i] - 0.035) ) * (1 / suitable_IO[i]) )
    return suitable_K[feature.index(max(feature))]






def myVMD(data,fs, ALPHA, K,TAU=0.00001, DC=0, INIT=0, TOL=1e-7):
    '''
    对VMD函数做了封装，抛弃了暂时不用的VMD函数的结果u_hat，简化了omega，并新增了corr和IO的计算
    :param data: 待VMD分解的数据
    :param fs: 信号的采样率
    :param ALPHA: 惩罚项alpha，重要参数
    :param K: 分解的模态数K，重要参数
    :param TAU: 对噪声的容忍，一般参数
    :param DC: 是否有直流项，一般参数
    :param INIT: 初始化频率分布，一般参数
    :param TOL:
    :return:
        u: 划分的结果，各IMF的集合
        omega: 各IMF的频率
        corr: 各IMF与原始信号的相关性，用作后续对IMF分量进行筛选
        IO: 本次VMD分解的正交性指数，用以判断本次VMD的好坏
    '''
    u, u_hat, omega = VMD(f=data, alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)
    # 只留下最终的omega
    omega = omega[-1, :] * fs
    # 计算各IMF与原始信号的相干性
    corr = []
    for i in range(K):
        if len(data) != len(u[i,:]):
            data = data[ : min(len(data), len(u[i,:]))]
            u[i,:] = u[i,:][ : min(len(data), len(u[i,:]))]

        corr.append(pearsonr(data, u[i, :])[0])
    # 计算本次VMD的IO值
    IOvalue = IO(data, u)

    return u, omega, corr, IOvalue

def IO(data, u):
    '''
    该函数计算了VMD的正交性系数index of orthogonality(IO), 其计算公式可见EMD的论文The empirical mode decomposition and the Hilbert spectrum for nonlinear and
    non-stationary time series analysis
    注意这里有一个坑：计算两两各IMF分量时需要保证 j != k， 这一点在原论文的公式中没有体现！
    :param data: 原始信号
    :param u: 分解后的各分量的集合，u的每一行都是一个IMF，与data同长度
    :return: 这个分解的IO值
    '''
    T = len(data)
    K = u.shape[0]

    sum = 0
    for t in range(T):
        sum += data[t]**2

    IO = 0
    for t in range(T):
        for j in range(K):
            for k in range(j + 1, K):
                    IO += u[j, :][t] * u[k, :][t] / sum
    return IO