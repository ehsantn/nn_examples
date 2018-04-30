import pandas as pd
import numpy as np
import h5py
import glob


def convert_images():
    image_dir = "/home/etotoni/pse-hpc/python/datasets/KITTI/training/image_2/*.png"
    files = sorted(glob.glob(image_dir))
    n_files = len(files)

    all_data = np.array([], dtype=np.uint8)
    data_ptr = 0
    offsets = np.empty(n_files + 1, dtype=np.uint64)

    for i in range(n_files):
        offsets[i] = data_ptr
        fname = files[i]
        data = np.fromfile(fname, dtype=np.uint8)
        all_data = np.concatenate((all_data, data))
        data_ptr += len(data)

    offsets[n_files] = data_ptr
    f = h5py.File("kitt_images.h5", "w")
    dset = f.create_dataset("img_data", all_data.shape, dtype=all_data.dtype)
    dset[:] = all_data
    dset = f.create_dataset("img_offsets", offsets.shape, dtype=offsets.dtype)
    dset[:] = offsets
    f.close()


if __name__ == '__main__':
    convert_images()
