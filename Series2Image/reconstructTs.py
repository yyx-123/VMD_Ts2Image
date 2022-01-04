import pickle
import matplotlib.pyplot as plt

with open('../dataset/images/SubImages/9/linear_32.pickle', 'rb') as f:
    images = pickle.load(f)

img = images[0]
LSF = img[0][0]
len = LSF.size()[0]
Ts = []
for i in range(len):
    Ts.append(LSF[i][i] / 2)

print(123)