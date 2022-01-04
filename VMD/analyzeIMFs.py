# 该脚本的作用是分析计算得到的IMF，可以反映出哪些被试，哪些通道在完成哪些任务时有比较好的表现
import os
import numpy as np

IMFDir = '../dataset/IMFs/'

totalRst = []
task1Cnt = []
task2Cnt = []
task3Cnt = []
for SubId in range(1, 31):
    Dir = IMFDir + str(SubId) + '/'
    task1ChannelCnt = [0] * 20
    task2ChannelCnt = [0] * 20
    task3ChannelCnt = [0] * 20
    SubRst = [task1ChannelCnt, task2ChannelCnt, task3ChannelCnt]
    for file in os.listdir(Dir):
        info = file.split('.')[0]
        task = int(info.split('_')[1].split('=')[1])
        channel = int(info.split('_')[2].split('=')[1])

        SubRst[task - 1][channel] += 1

    print('SubID:{}'.format(SubId))
    print(SubRst)
    task1Cnt.append(np.sum(SubRst[0]))
    task2Cnt.append(np.sum(SubRst[1]))
    task3Cnt.append(np.sum(SubRst[2]))
    # print('{}, {}, {}'.format(task1Cnt[-1], task2Cnt[-1], task3Cnt[-1]))
    totalRst.append(SubRst)

print(np.sum(task1Cnt))
print(np.sum(task2Cnt))
print(np.sum(task3Cnt))