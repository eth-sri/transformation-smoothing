import torch.multiprocessing as mp
from typing import *
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformations import *
from operator import itemgetter
import PIL
import os
import numpy as np
import copy
import math
import robustness, robustness.datasets, robustness.tools
from argparse import Namespace
import random

IMAGENET_PATH = os.path.abspath("./ds/imagenet/")
GTSRB_PATH = os.path.abspath("./ds/imagenet/")

def get_dataset(args, dataset_split, transform=None):
    if args.dataset == 'mnist':
        return datasets.MNIST('./ds/mnist', train=(dataset_split == 'train'),
                              download=True, transform=transform)
    elif args.dataset == 'fashionmnist':
        return datasets.FashionMNIST('./ds/fmnist', train=(dataset_split == 'train'),
                                     download=True, transform=transform)
    elif args.dataset == 'cifar':
        return datasets.CIFAR10('./ds/cifar', train=(dataset_split == 'train'),
                                download=True, transform=transform)
    elif args.dataset == 'imagenet':
        return datasets.ImageFolder(root=os.path.join(os.path.abspath("./ds/imagenet/"), dataset_split),
                                    transform=transform)
    elif args.dataset == 'restricted_imagenet':
        ds = robustness.datasets.RestrictedImageNet(os.path.abspath("./ds/imagenet/"))
        test_path = os.path.join(os.path.abspath("./ds/imagenet/"), 'val')
        return robustness.tools.folder.ImageFolder(root=test_path,
                                                   label_mapping=ds.label_mapping, transform=transform)
    elif args.dataset == 'GTSRB':
        path = os.path.abspath("./ds/GTSRB")
        if "gtsrb_path" in args:
            path = args.gtsrb_path
        return datasets.ImageFolder(root=os.path.join(path, dataset_split), transform=transform)
    else:
        assert(False)


# def get_path_label(path):
#     return path


# def get_size(path):
#     with open(path, 'rb') as f:
#         img = PIL.Image.open(f, mode='r')
#         return img.size


class ImagenetSizeDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_split, min_size, I, sizes):
        assert(dataset_split in ['train', 'val', 'test']) 
        self.data = datasets.ImageFolder(root=os.path.join(os.path.abspath("./ds/imagenet/"), dataset_split))
        self.I = I
        self.sizes = sizes
        self.min_size = min_size

    def __getitem__(self, index):
        return self.data[self.I[index]]

    def __len__(self):
        return len(self.I)
    
    
class ImagenetSizeLoader:

    def __init__(self, n_threads, dataset_split):
        assert(dataset_split in ['train', 'val', 'test']) 
        self.dataset_split = dataset_split
        self.n_threads = n_threads
        with mp.Pool(self.n_threads) as pool:
            data = datasets.ImageFolder(root=os.path.join(os.path.abspath("./ds/imagenet/"), dataset_split), loader=get_path_label)
            self.paths, self.labels = zip(*data)
            self.sizes = pool.map(get_size, self.paths)
        
    def get_indices(self, min_size):
        I = map(lambda x: min(x) > min_size, self.sizes)
        I = filter(itemgetter(1), enumerate(I))
        I = map(itemgetter(0), I)
        return list(I)

    def get_dataset(self, min_size=0):
        I = self.get_indices(min_size)
        sizes = [self.sizes[i] for i in I]
        return ImagenetSizeDataset(self.dataset_split, min_size, I, sizes)
