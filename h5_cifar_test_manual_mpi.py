import numpy as np
import h5py
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torch.utils.data
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD
node_id = comm.Get_rank()
num_pes = comm.Get_size()
batch_size = 4 // num_pes

def dist_get_start(total):
    div_chunk = int(math.ceil(total / num_pes));
    start = min(total, node_id * div_chunk);
    return start

def dist_get_end(total):
    div_chunk = int(math.ceil(total / num_pes));
    end = min(total, (node_id+1) * div_chunk);
    return end

def average_gradients(model):
    for param in model.parameters():
        source = param.grad.data.numpy()
        dest = np.empty_like(source)
        comm.Allreduce(source, dest, op=MPI.SUM)
        # print(node_id, source, dest)
        param.grad.data = torch.Tensor(dest) / num_pes

class H5_dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.train = train
        file_name = "cifar.hdf5"
        f = h5py.File(file_name, "r")
        if train:
            total = len(f["train_data"])
            start = dist_get_start(total)
            end = dist_get_end(total)
            # print(total, start, end)
            self.train_data = f["train_data"][start:end]
            # convert from [0,255] to [0.0,1.0]
            self.train_data = (self.train_data) / np.float32(255.0)
            self.train_data = (self.train_data - np.float32(0.5)) / np.float32(0.5)
            # convert to CHW
            # self.train_data = self.train_data.transpose((0, 2, 3, 1))
            self.train_labels = f["train_labels"][start:end]
        else:
            total = len(f["test_data"])
            start = dist_get_start(total)
            end = dist_get_end(total)
            self.test_data = f["test_data"][start:end]
            # convert from [0,255] to [0.0,1.0]
            self.test_data = (self.test_data) / np.float32(255.0)
            self.test_data = (self.test_data - np.float32(0.5)) / np.float32(0.5)
            # convert to CHW
            # self.test_data = self.test_data.transpose((0, 2, 3, 1))
            self.test_labels = f["test_labels"][start:end]
        f.close()
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], int(self.train_labels[index])
        else:
            img, target = self.test_data[index], int(self.test_labels[index])
        # p_img = Image.fromarray(img)
        return img, target
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

trainloader = torch.utils.data.DataLoader(H5_dataset(train=True), batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(H5_dataset(train=False), batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        average_gradients(net)
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
