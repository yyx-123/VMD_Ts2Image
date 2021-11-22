import numpy as np
from scipy.stats import pearsonr
from math import pi as PI
import matplotlib.pyplot as plt
from vmdpy import VMD

# 构造测试信号
fs = 1000
t = np.linspace(0,2,fs * 2,endpoint=True)

f1 = 2
f2 = 24
f3 = 288
A1 = 1
A2 = 0.25
A3 = 1/8

v1 = A1 * np.cos(2 * PI * f1 * t)
v2 = A2 * np.cos(2 * PI * f2 * t)
v3 = A3 * np.cos(2 * PI * f3 * t)
x = np.concatenate((v1[:601], v2[601:1301], v3[1301:])) + 0.05 * np.random.randn(2000)
# plt.subplot(2,1,1)
# plt.plot(t, x)

#. some sample parameters for VMD
ALPHA = fs       # moderate bandwidth constraint
TAU = 0.            # noise-tolerance (no strict fidelity enforcement)
K = 5              # 3 modes
DC = 0             # no DC part imposed
INIT = 1           # initialize omegas uniformly
TOL = 1e-7
#. Run actual VMD code
u, u_hat, omega = VMD(f=x, alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)

# 检验VMD结果
iters = omega.shape[0]      # iters反映了算法迭代至稳定的迭代数。VMD源码中规定的迭代数量上限是500，因此需要检查该iters是否大于500
print(iters)
if iters < 500:
    print(omega[-1, : ] * fs)

for i in range(K):
    plt.subplot(K, 2, 2 * i + 1)
    plt.plot(t, u[i,:])
    plt.subplot(K, 2, 2 * i + 2)
    plt.plot(np.fft.fft(u[i, :]))
    print(pearsonr(x, u[i, :])[0])
plt.show()

