import numpy as np
import hpat
import h5py

@hpat.jit
def convert_data():
    fname = "/export/intel/users/etotoni/BXP5401-front-camera_2017.dat"
    # fname = "img2.dat"
    blob = np.fromfile(fname, np.uint8)

    # reshape to images
    n_channels = 3
    height = 800
    width = 1280
    n_images = len(blob)//(n_channels*height*width)
    data = blob.reshape(n_images, height, width, n_channels)

    file_name = "/export/intel/users/etotoni/BXP5401-data.hdf5"
    # file_name = "img2.hdf5"
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("front_cam", (n_images, height, width, n_channels), dtype='i1')
    dset1[:] = data
    f.close()
    return

if __name__ == '__main__':
    convert_data()
