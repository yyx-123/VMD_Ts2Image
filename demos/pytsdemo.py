import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import VMD.utils as utils
from tqdm import tqdm
import scipy.io as scio
import time
import HRF


# 提取任务数据
taskdata = scio.loadmat("D:\\fnirs dataset\left right foot tapping\\taskInfo.mat")
triggers = taskdata['triggers']     # 每一行是一个被试的任务开始采样点编号，30*75
event = taskdata['event'][0]           # 1 2 3 3 2 1 ... 的序列，描述了任务性质
event_desc = taskdata['event_desc'][0] # 对1、2、3的描述
LEN = 400
Ts = 0.075
fs = 1 / Ts
t = np.linspace(0, Ts*LEN, LEN)

ALPHA = 150
for SubId in range(17, 18):
    nirsdata = scio.loadmat('D:\\fnirs dataset\left right foot tapping\\2_0preprocessedData\\Sub' + str(SubId) + '.mat')['nirsdata'][0][0][0]
    for taskNum in range(10, 30):
        startpoint = triggers[SubId - 1][taskNum]
        for channel in range(15, 16):
            data = nirsdata[:, channel][startpoint: startpoint + LEN]

            K = utils.findBestK(data=data, fs=fs, ALPHA=ALPHA, Kmin=2, Kmax=8)
            if K > 0:
                u, omega, corr, IO = utils.myVMD(data=data, fs=fs, ALPHA=ALPHA, K=K)
                data = u[np.argmin(omega),:].reshape(1, -1)

                image_size = 64
                gasf = GramianAngularField(image_size=image_size, method='summation')
                data_gasf = gasf.fit_transform(data)
                gadf = GramianAngularField(image_size=image_size, method='difference')
                data_gadf = gadf.fit_transform(data)
                mtf = MarkovTransitionField(image_size=image_size, n_bins=32)
                data_mtf = mtf.fit_transform(data)
                images = [data_gasf[0], data_gadf[0], data_mtf[0]]
                titles = ['Summation', 'Difference', 'MTF']

                fig, axs = plt.subplots(1, 3, constrained_layout=True)
                for image, title, ax in zip(images, titles, axs):
                    if title == 'MTF':
                        ax.imshow(image, cmap='rainbow', origin='lower', vmin=0., vmax=1.)
                        ax.set_title(title)
                    else:
                        ax.imshow(image, cmap='rainbow',origin='lower', vmin=-1., vmax=1.)
                        ax.set_title(title)
                fig.suptitle('SubId={}, taskNum={}, task={}\n channel={}, omega={:3f}'.format(SubId, taskNum, event_desc[event[taskNum] - 1], channel, omega[0]), fontsize=16)
                plt.margins(0, 0)
                # 保存图片
                # plt.savefig("GramianAngularField.pdf", pad_inches=0)
                plt.show()




