import numpy as np
import glob
import hpat
from hpat import prange
import h5py
import time
import cv2
# cv2.setNumThreads(0)

#@hpat.jit
def load_images():
    fname = "kitt_images.h5"
    f = h5py.File(fname)
    img_offsets = f['img_offsets'][:]
    n_imgs = len(img_offsets)-1
    dataset = np.empty((n_imgs, 360, 1000, 3), np.uint8)
    t1 = time.time()
    for i in prange(n_imgs):
        len_img = img_offsets[i+1] - img_offsets[i]
        img_dat = f['img_data'][img_offsets[i]:img_offsets[i+1]]
        img = cv2.imdecode(img_dat, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (1000, 360))
        dataset[i] = img_resized

    m = dataset.mean()
    print("mean ", m, "\nExec time", time.time()-t1);

load_images()
