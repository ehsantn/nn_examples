import torchvision
import os, pickle
import numpy as np

def create_dataset():
    trainset = torchvision.datasets.CIFAR10(root='./data', download=True)
    fname = "./data/cifar-10-batches-py/data_batch_1"
    fo = open(fname, 'rb')
    entry = pickle.load(fo, encoding='latin1')
    train_data = entry['data']
    fo.close()
    train_data.tofile("train_data.dat")

create_dataset()

import time
import hpat
from hpat import prange
import cv2
hpat.multithread_mode = True
cv2.setNumThreads(0)  # we use threading across images

@hpat.jit(locals={'images:return': 'distributed'})
def read_data():
    file_name = "train_data.dat"
    blob = np.fromfile(file_name, np.uint8)
    # reshape to images
    n_channels = 3
    height = 32
    width = 32
    n_images = len(blob)//(n_channels*height*width)
    data = blob.reshape(n_images, height, width, n_channels)

    # resize
    resize_len = 224
    images = np.empty((n_images, resize_len, resize_len, n_channels), np.uint8)
    for i in prange(n_images):
        images[i] = cv2.resize(data[i], (resize_len, resize_len))

    # convert from [0,255] to [0.0,1.0]
    # normalize
    u2f_ratio = np.float32(255.0)
    c0_m = np.float32(0.485)
    c1_m = np.float32(0.456)
    c2_m = np.float32(0.406)
    c0_std = np.float32(0.229)
    c1_std = np.float32(0.224)
    c2_std = np.float32(0.225)
    for i in prange(n_images):
        images[i,:,:,0] = (images[i,:,:,0]/ u2f_ratio - c0_m) / c0_std
        images[i,:,:,1] = (images[i,:,:,1]/ u2f_ratio - c1_m) / c1_std
        images[i,:,:,2] = (images[i,:,:,2]/ u2f_ratio - c2_m) / c2_std

    # convert to CHW
    images = images.transpose(0, 3, 1, 2)
    return images

t1 = time.time()
imgs = read_data()
#hpat.distribution_report()
print("data read time", time.time()-t1)

from torch import Tensor
from torch.autograd import Variable
model = torchvision.models.resnet18(True)
t1 = time.time()
res = model(Variable(Tensor(imgs[:100])))
print("dnn time", time.time()-t1)

# get top class stats
vals, inds = res.max(1)

import pandas as pd

@hpat.jit(locals={'vals:input': 'distributed', 'inds:input': 'distributed'})
def get_stats(vals, inds):
    df = pd.DataFrame({'vals': vals, 'classes': inds})
    stat = df.describe()
    print(stat)
    TRUCK = 717
    print((inds == TRUCK).sum())

get_stats(vals.data.numpy(), inds.data.numpy())
#hpat.distribution_report()
