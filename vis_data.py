import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

fname = "/Users/etotoni/cars.dat"
blob = np.fromfile(fname, np.uint8)
# reshape to images
n_channels = 3
height = 600
width = 600
n_images = len(blob)//(n_channels*height*width)
data = blob.reshape(n_images, height, width, n_channels)

plt.imshow(data[0])
plt.xticks([]), plt.yticks([])
plt.show()
