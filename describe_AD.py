import numpy as np
import torch
import torchvision
import cv2
from torch.autograd import Variable
import time
import pandas as pd
import hpat
from hpat import prange
hpat.multithread_mode = True
cv2.setNumThreads(0)  # we use threading across images

model = torchvision.models.resnet18(True)

@hpat.jit(locals={'fdata:return': 'distributed'})
def read_data():
    #fname = "/export/intel/lustre/etotoni/BXP5401-front-camera_2017.dat"
    fname = "img2.dat"
    blob = np.fromfile(fname, np.uint8)

    # reshape to images
    n_channels = 3
    height = 800
    width = 1280
    n_images = len(blob)//(n_channels*height*width)
    data = blob.reshape(n_images, height, width, n_channels)

    # select every 5 image
    data = data[::5,:,:,:]
    n_images = len(data)

    # crop to 600 by 600
    data = data[:,100:-100, 340:-340,:]

    # resize
    resize_len = 224
    resized_images = np.empty((n_images, resize_len, resize_len, n_channels), np.uint8)
    for i in prange(n_images):
        resized_images[i] = cv2.resize(data[i], (resize_len, resize_len))

    # convert from [0,255] to [0.0,1.0]
    fdata = (resized_images) / np.float32(255.0)
    fdata = (fdata - np.float32(0.5)) / np.float32(0.5)

    # convert to CHW
    fdata = fdata.transpose((0, 3, 1, 2))
    return fdata


images = read_data()
#hpat.distribution_report()

# convert to Tensor
imgs_tensor = torch.Tensor(images)

t1 = time.time()
res = model(Variable(imgs_tensor))
print("Model evaluation time", time.time()-t1)

# get stats
vals, inds = res.max(1)

@hpat.jit(locals={'vals:input': 'distributed', 'inds:input': 'distributed'})
def get_stats(vals, inds):
    df = pd.DataFrame({'vals': vals, 'classes': inds})
    s1 = df.vals.describe()
    print(s1)
    s2 = df.classes.describe()
    print(s2)

get_stats(vals.data.numpy(), inds.data.numpy())
#hpat.distribution_report()
