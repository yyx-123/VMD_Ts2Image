import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

ts1 = np.array([5,5,5,5,5,5,5,5,5,5,5,6,7,8,9,10,9,8,7,6,5]).reshape(1, -1)
ts2 = np.array([5,6,7,8,9,10,9,8,7,6,5,5,5,5,5,5,5,5,5,5,5]).reshape(1, -1)

image_size = 20
gasf = GramianAngularField(image_size=image_size, method='summation')
ts1_gasf = gasf.fit_transform(ts1)
ts2_gasf = gasf.fit_transform(ts2)
gadf = GramianAngularField(image_size=image_size, method='difference')
ts1_gadf = gadf.fit_transform(ts1)
ts2_gadf = gadf.fit_transform(ts2)

images = [ts1_gasf[0], ts2_gasf[0], ts1_gadf[0], ts2_gadf[0]]
titles = ['ts1_gasf', 'ts2_gasf', 'ts1_gadf', 'ts2_gadf']

fig, axs = plt.subplots(1, 4, constrained_layout=True)
for image, title, ax in zip(images, titles, axs):
    ax.imshow(image, cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
    ax.set_title(title)
plt.show()