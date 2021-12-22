import numpy as np
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import MarkovTransitionField



class TimeSeriesTransformer():
    '''
    该transformer类的作用是将一个时间序列 ts 转化成一个尺寸在 imageSize * imageSize 的 image
    '''
    def __init__(self, ts, imageSize):
        self.ts = ts
        self.imageSize = imageSize

    def normalizeAndCompactTs(self, lowerBound, higherBound):
        '''
        将时间序列 ts 缩放到 [lowerBound, highBound] 区间内，并用 PAA 的方法降采样，最后返回 rescaled_ts 的列向量表达
        :param lowerBound:
        :param higherBound:
        :return:
        '''
        ts = self.ts.reshape(1, -1)
        min_ts, max_ts = np.min(ts), np.max(ts)
        factor = (higherBound - lowerBound) / (max_ts - min_ts)
        rescaled_ts = lowerBound + factor * (ts - min_ts)

        paa = PiecewiseAggregateApproximation(window_size=None, output_size=self.imageSize, overlapping=False)
        rescaled_ts = paa.fit_transform(rescaled_ts)

        return rescaled_ts.reshape(-1, 1)

    def GAF(self, lowerBound=0, higherBound=1):
        # step 1、2: 标准化时间序列、PAA压缩长度
        rescaled_ts = self.normalizeAndCompactTs(lowerBound=lowerBound, higherBound=higherBound)

        # step 3: 计算GAF
        sin_ts = np.sqrt(np.clip(1 - rescaled_ts ** 2, 0, 1))
        GASF = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
        GADF = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)

        # step4: 返回结果
        return GASF, GADF

    def MTF(self, nBins=32):
        mtfModel = MarkovTransitionField(image_size=self.imageSize, n_bins=nBins)
        MTF = mtfModel.fit_transform(self.ts)[0]

        return MTF

    def linear(self, lowerBound=-0.5, higherBound=0.5):
        # step 1、2: 标准化时间序列、PAA压缩长度
        x = self.normalizeAndCompactTs(lowerBound=lowerBound, higherBound=higherBound)
        I = np.ones(len(x)).reshape(-1, 1)

        nonMappingSum = np.outer(x, I) + np.outer(I, x)
        nonMappingDiff = np.outer(x, I) - np.outer(I, x)

        return nonMappingSum, nonMappingDiff

    def sigmoidMapping(self, lowerBound=0.001, higherBound=0.999):
        # step 1、2: 标准化时间序列、PAA压缩长度
        x = self.normalizeAndCompactTs(lowerBound=lowerBound, higherBound=higherBound)
        I = np.ones(len(x)).reshape(-1, 1)

        sigmoidMappingSum = np.outer(x, x) / (np.outer(I, I) - np.outer(x, I) - np.outer(I, x) + 2 * np.outer(x, x))
        sigmoidMappingDiff = (np.outer(x, I) - np.outer(x, x)) / (np.outer(x, I) + np.outer(I, x) - 2 * np.outer(x, x))

        return sigmoidMappingSum, sigmoidMappingDiff

    def tanMapping(self, lowerBound=-0.999, higherBound=0.999):
        # step 1、2: 标准化时间序列、PAA压缩长度
        x = self.normalizeAndCompactTs(lowerBound=lowerBound, higherBound=higherBound)
        I = np.ones(len(x)).reshape(-1, 1)

        tanMappingSum = (np.outer(x, I) + np.outer(I, x)) / (np.outer(I, I) - np.outer(x, x))
        tanMappingDiff = (np.outer(x, I) - np.outer(I, x)) / (np.outer(I, I) + np.outer(x, x))

        return tanMappingSum, tanMappingDiff

    def tanhMapping(self, lowerBound=-0.999, higherBound=0.999):
        # step 1、2: 标准化时间序列、PAA压缩长度
        x = self.normalizeAndCompactTs(lowerBound=lowerBound, higherBound=higherBound)
        I = np.ones(len(x)).reshape(-1, 1)

        tanhMappingSum = (np.outer(x, I) + np.outer(I, x)) / (np.outer(I, I) + np.outer(x, x))
        tanhMappingDiff = (np.outer(x, I) - np.outer(I, x)) / (np.outer(I, I) - np.outer(x, x))

        return tanhMappingSum, tanhMappingDiff

    def visualize(self, mode='Sum'):
        plt.subplot(3,2,1)
        plt.plot(self.ts[0])
        plt.title('original time seires')

        GASF, GADF = self.GAF()
        nonMappingSum, nonMappingDiff = self.linear()
        sigmoidSum, sigmoidDiff = self.sigmoidMapping()
        tanSum, tanDiff = self.tanMapping()
        tanhSum, tanhDiff = self.tanhMapping()

        if mode == 'Sum':
            plt.suptitle('Sum mode')

            plt.subplot(3,2,2)
            plt.imshow(GASF)
            plt.title('GAF')

            plt.subplot(3,2,3)
            plt.imshow(nonMappingSum)
            plt.title('nonMapping')

            plt.subplot(3,2,4)
            plt.imshow(sigmoidSum)
            plt.title('sigmoid')

            plt.subplot(3,2,5)
            plt.imshow(tanSum)
            plt.title('tan')

            plt.subplot(3,2,6)
            plt.imshow(tanhSum)
            plt.title('tanh')
        elif mode == 'Diff':
            plt.suptitle('Diff mode')

            plt.subplot(3, 2, 2)
            plt.imshow(GADF)
            plt.title('GAF')

            plt.subplot(3, 2, 3)
            plt.imshow(nonMappingDiff)
            plt.title('nonMapping')

            plt.subplot(3, 2, 4)
            plt.imshow(sigmoidDiff)
            plt.title('sigmoid')

            plt.subplot(3, 2, 5)
            plt.imshow(tanDiff)
            plt.title('tan')

            plt.subplot(3, 2, 6)
            plt.imshow(tanhDiff)
            plt.title('tanh')

        plt.show()


if __name__=='__main__':
    x = np.array([1, 2, 3, 4, 5])
    transformer = TimeSeriesTransformer(ts=x, imageSize=5)
    transformer.visualize(mode='Diff')



