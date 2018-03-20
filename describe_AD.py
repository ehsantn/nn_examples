import numpy as np
import torch
import torchvision
import cv2
from torch.autograd import Variable
import time
import pandas as pd
import hpat
hpat.multithread_mode = True

model = torchvision.models.resnet18(True)

@hpat.jit(locals={'images:return': 'distributed'})
def read_data():
    #fname = "/export/intel/lustre/etotoni/BXP5401-front-camera_2017.dat"
    fname = "img2.dat"
    blob = np.fromfile(fname, np.uint8)
    n_channels = 3
    height = 800
    width = 1280
    n_images = len(blob)//(n_channels*height*width)
    data = blob.reshape(n_images, height, width, n_channels)[::100,:,:,:]
    # convert from [0,255] to [0.0,1.0]
    fdata = (data) / np.float32(255.0)
    fdata = (fdata - np.float32(0.5)) / np.float32(0.5)
    # convert to CHW
    # data = data.transpose((0, 2, 3, 1))
    return fdata

images = read_data()
print(images.sum())
n_images = len(images)
n_channels = 3
resize_len = 224
# resize
resized_images = np.empty((n_images, resize_len, resize_len, n_channels), np.float32)
for i in range(n_images):
    resized_images[i] = cv2.resize(images[i], (resize_len, resize_len))

# convert to CHW Tensor
imgs_tensor = torch.Tensor(resized_images.transpose((0,3,1,2)))

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
