import numpy as np
import glob
import hpat
from hpat import prange
import time
import cv2
# cv2.setNumThreads(0)

@hpat.jit
def load_images():
    files = glob.glob("/home/etotoni/pse-hpc/python/datasets/KITTI/training/image_2/007*png")
    n = len(files)
    dataset = np.empty((n, 360, 1000, 3), np.uint8)
    t1 = time.time()
    for i in prange(n):
        f = files[i]
        img = cv2.imread(f)
        img_resized = cv2.resize(img, (1000, 360))
        dataset[i] = img_resized

    m = dataset.mean()
    print("mean ", m, "\nExec time", time.time()-t1);

load_images()
