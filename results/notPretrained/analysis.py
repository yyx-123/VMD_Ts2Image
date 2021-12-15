import numpy as np

features = ['MTF', 'GAF', 'GAF_MTF',  'GADF_MTF', 'GASF_MTF', 'nonMapping', 'MTF_nonMapping', 'sigmoid','MTF_sigmoid','tan', 'MTF_tan', 'tanh', 'MTF_tanh']
sizes = [32, 64]

dict = {}
for feature in features:
    avgAcc = -1
    for size in sizes:
        accs = []
        with open(feature + '_' + str(size) + '.txt') as result:
            lines = result.readlines()
            for i in range(70, 80):
                idx = lines[i].find('acc:')
                accs.append(float(lines[i][idx + 4: idx + 10]))
        avgAcc = max(avgAcc, np.mean(accs))
    dict.update({feature : avgAcc})

print(dict)

