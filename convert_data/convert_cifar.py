import pickle
import os
import numpy as np
import h5py

base_folder = "./data/cifar-10-batches-py/"
file_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    ]

file_test_list = [
        'test_batch',
    ]

train_data = []
train_labels = []

# load training data
for f in file_list:
    fname = os.path.join(base_folder, f)
    fo = open(fname, 'rb')
    entry = pickle.load(fo, encoding='latin1')
    train_data.append(entry['data'])
    train_labels += entry['labels']
    fo.close()

train_data = np.concatenate(train_data)
train_data = train_data.reshape((50000, 3, 32, 32))
#train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC

# load test data
fname = os.path.join(base_folder, file_test_list[0])
fo = open(fname, 'rb')
entry = pickle.load(fo, encoding='latin1')
test_data = entry['data']
test_labels = entry['labels']
fo.close()

test_data = test_data.reshape((10000, 3, 32, 32))
#test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC

# write hdf5 file
file_name = "cifar.hdf5"
f = h5py.File(file_name, "w")
dset1 = f.create_dataset("train_data", train_data.shape, dtype='u1')
dset1[:] = train_data

tl_arrays = np.array(train_labels, dtype=np.uint8)
ldset1 = f.create_dataset("train_labels", tl_arrays.shape, dtype='u1')
ldset1[:] = tl_arrays


dset2 = f.create_dataset("test_data", test_data.shape, dtype='u1')
dset2[:] = test_data

ttl_arrays = np.array(test_labels, dtype=np.uint8)
ldset2 = f.create_dataset("test_labels", ttl_arrays.shape, dtype='u1')
ldset2[:] = ttl_arrays

f.close()
