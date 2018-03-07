import pandas as pd
import numpy as np
import h5py
import glob


fname = "/home/etotoni/pse-hpc/python/datasets/KITTI/training/label_2/006655.txt"
dtype_dict = {'type': str, 'truncated': np.float32,
               'occluded': np.uint8, 'alpha': np.float32,
               'bbox_l': np.float32, 'bbox_t': np.float32,
               'bbox_r': np.float32, 'bbox_b': np.float32,
               'dim_h': np.float32, 'dim_w': np.float32,
               'dim_l': np.float32, 'loc_x': np.float32,
               'loc_y': np.float32, 'loc_z': np.float32,
               'rot_y': np.float32}


df = pd.read_csv(fname, sep=' ', names=list(dtype_dict.keys()),
                 dtype=dtype_dict)
type_convert = {'Car': 0, 'Van': 1, 'Truck': 2,
                     'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6,
                     'Misc': 7, 'DontCare': 8}

df['type'] = df['type'].apply(lambda a: type_convert[a])
dtype_dict['type'] = np.uint8

N = len(df)
f = h5py.File("kitt_labels.h5", "w")
for name, dt in dtype_dict.items():
    dset = f.create_dataset(name, (N,), dtype=dt)
    dset[:] = df[name]

f.close()
print(df)
