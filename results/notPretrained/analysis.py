import numpy as np

features = ['MTFnBINS8', 'MTFnBINS16','MTFnBINS32', 'MTFnBINS64']
sizes = [16, 32, 64]

dict = {}
for feature in features:
    yyy = []
    for size in sizes:
        accs = []
        with open(feature + '_' + str(size) + '.pickle.txt') as result:
            lines = result.readlines()
            for i in range(70, 80):
                idx = lines[i].find('acc:')
                accs.append(float(lines[i][idx + 4: idx + 10]))
        acc = np.mean(accs)
        yyy.append(acc)
    print(yyy)
    # dict.update({feature : yyy})

print(dict)

