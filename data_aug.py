
import paddle
import paddle.vision.transforms as transforms
import paddle.io as data
from randaugment import *


data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'fmnist': ((0.2860,), (0.3530,)),
              'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687))}




class MixDataset(data.Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = paddle.randint(0, len(self.dataset), (1,)).item()
        x,label = self.dataset.data[index]
        input = {'data': x, 'target': label}
        return input

    def __len__(self):
        return self.size


