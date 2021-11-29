import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import utils

# 提取任务数据和近红外数据
taskdata = scio.loadmat("D:\\fnirs dataset\left right foot tapping\\taskInfo.mat")
triggers = taskdata['triggers']     # 每一行是一个被试的任务开始采样点编号，30*75
event = taskdata['event']           # 1 2 3 3 2 1 ... 的序列，描述了任务性质
event_desc = taskdata['event_desc'] # 对1、2、3的描述
nirsdata = scio.loadmat('D:\\fnirs dataset\left right foot tapping\\2_0preprocessedData\\Sub17.mat')['nirsdata'][0][0][0]
LEN = 400
channel = 15
startpoint = triggers[16][35]
# 提取出第channel通道（channel=0,1,2,3..19）的第startpoint起长度为Len的数据。startpoint和Len都是以采样点为单位。因此一个task30秒的Len约为400
data = nirsdata[:, channel][startpoint: startpoint + LEN]
Ts = 0.075
fs = 1 / Ts
t = np.linspace(0, Ts*LEN, LEN)



#. some sample parameters for VMD
ALPHA = 150


#. Run actual VMD code
K = utils.findBestK(data=data, fs=fs, ALPHA=ALPHA, Kmin=2, Kmax=8)
if K > 0:
    plt.subplot(K + 1, 1, 1)
    plt.plot(t, data)

    u, omega, corr, IO = utils.myVMD(data=data, fs=fs, ALPHA=ALPHA, K=K)
    print("omega =", omega)
    print("corr =", corr)
    print("IO value =", IO)

    for i in range(K):
        plt.subplot(K + 1, 1, i + 2)
        plt.plot(t, u[i,:])
    plt.show()
