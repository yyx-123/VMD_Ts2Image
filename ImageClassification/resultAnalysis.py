import os
import numpy as np



features = ['GAF_MTF', 'MTF_nonMapping', 'MTF_sigmoid', 'MTF_tan', 'MTF_tanh']
imageSizes = [16, 32, 48, 64]
resultDir = '../results/'

comp = []
for feature in features:
    featureRst = []
    for imageSize in imageSizes:
        fileName = feature + '_' + str(imageSize) + '.txt'
        with open(resultDir + fileName, 'r') as result:
            lines = result.readlines()
            accs = []
            for i in range(91, 101):
                line = lines[i]
                idx = line.find('acc:')
                accs.append(float(line[idx + 4 : idx + 10]))
        featureRst.append(np.mean(accs))
    comp.append(featureRst)

for i in range(5):
    print(comp[i])
