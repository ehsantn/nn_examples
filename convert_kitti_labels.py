import pandas as pd
import numpy as np
import h5py
import glob

def convert_labels():
    dtype_dict = {'type': str, 'truncated': np.float32,
               'occluded': np.uint8, 'alpha': np.float32,
               'bbox_l': np.float32, 'bbox_t': np.float32,
               'bbox_r': np.float32, 'bbox_b': np.float32,
               'dim_h': np.float32, 'dim_w': np.float32,
               'dim_l': np.float32, 'loc_x': np.float32,
               'loc_y': np.float32, 'loc_z': np.float32,
               'rot_y': np.float32}
    label_dir = "/home/etotoni/pse-hpc/python/datasets/KITTI/training/label_2/*.txt"
    files = sorted(glob.glob(label_dir))
    n_files = len(files)
    all_data = pd.DataFrame()
    for i in range(n_files):
        fname = files[i]
        df = pd.read_csv(fname, sep=' ', names=list(dtype_dict.keys()),
                         dtype=dtype_dict)
        type_convert = {'Car': 0, 'Van': 1, 'Truck': 2,
                             'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6,
                             'Misc': 7, 'DontCare': 8}

        # convert string type
        df['type'] = df['type'].apply(lambda a: type_convert[a])
        df['img_id'] = np.full(len(df), i, dtype=np.uint32)
        all_data = all_data.append(df, ignore_index=True)

    dtype_dict['type'] = np.uint8
    dtype_dict['img_id'] = np.uint32
    N = len(all_data)
    f = h5py.File("kitt_labels.h5", "w")
    for name, dt in dtype_dict.items():
        dset = f.create_dataset(name, (N,), dtype=dt)
        dset[:] = all_data[name]

    f.close()


if __name__ == '__main__':
    convert_labels()
