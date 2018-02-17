import numpy as np
import h5py
from PIL import Image
import torch.utils.data

class H5_dataset(torch.utils.data.Dataset):
    def __init__(self):
        file_name = "cifar.hdf5"
        f = h5py.File(file_name, "r")
        self.train_data = f["train_data"][:]
        # convert from [0,255] to [0.0,1.0]
        self.train_data = (self.train_data) / np.float32(255.0)
        self.train_data = (self.train_data - np.float32(0.5)) / np.float32(0.5)
        # convert to CHW
        # self.train_data = self.train_data.transpose((0, 2, 3, 1))
        self.train_labels = f["train_labels"][:]
        f.close()
    def __getitem__(self, index):
        img, target = self.train_data[index], int(self.train_labels[index])
        #p_img = Image.fromarray(img)
        return img, target
    def __len__(self):
        return len(self.train_data)

trainloader = torch.utils.data.DataLoader(H5_dataset(), batch_size=4,
                                          shuffle=True, num_workers=2)
