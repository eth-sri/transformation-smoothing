from archs.cifar_resnet import resnet as resnet_cifar
from datasets import get_normalize_layer, get_input_center_layer
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import numpy as np
from PIL import Image, ImageDraw


def get_vingette_mask(shape, offset=0, shape_type='circ'):
    assert len(shape) == 3
    if shape_type == 'circ':
        c, h, w = shape
        assert h == w
        mask = Image.new('I', (h, w))
        draw = ImageDraw.Draw(mask)
        lu = (0+offset, 0+offset)
        rd = (h-offset, h-offset)
        draw.ellipse([lu, rd], fill=(1))
        del draw
        mask = np.array(mask).astype(np.float32)
        mask = np.tile(np.expand_dims(mask, axis=0), (c, 1, 1))
        return mask
    elif shape_type == 'rect':
        offset = int(offset)
        mask = np.ones(shape)
        mask[:, :offset, :]  = 0
        mask[:, -offset:, :]  = 0
        mask[:, :, -offset:]  = 0
        mask[:, :, :offset]  = 0
        return mask


class VingetteModule(nn.Module):
    def __init__(self, size, shape, offset):
        super().__init__()
        V = get_vingette_mask(size,
                              shape_type=shape,
                              offset=offset)
        V = torch.tensor(V, dtype=torch.float).unsqueeze(0)
        self.V = nn.Parameter(V, requires_grad=False)
    def forward(self, x):
        return self.V * x

def getGaussianFilter(filter_size, sigma):
    assert(filter_size % 2 == 1)
    G = np.zeros((filter_size, filter_size))
    c = filter_size // 2
    for i in range(c + 1):
        for j in range(c + 1):
            r = i * i + j * j
            G[c + i, c + j] = r
            G[c + i, c - j] = r
            G[c - i, c + j] = r
            G[c - i, c - j] = r
    G = np.exp(-G / sigma) / (np.pi * sigma)
    G = G / G.sum()
    return G


class Filter(nn.Module):
    def __init__(self, size, sigma, channels):
        super().__init__()
        self.channels = channels
        self.size = size
        self.c = size // 2
        self.sigma = sigma
        G = getGaussianFilter(size, sigma)
        G = np.repeat(G[np.newaxis, np.newaxis, :], self.channels, 0)
        G = torch.tensor(G, dtype=torch.float)
        self.G = nn.Parameter(G, requires_grad=False)

    #  so I can also use it as a transformer
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        shape = x.size()
        squeeze = False
        if len(shape) == 4:
            b, c, h, w = shape
        elif len(shape) == 3:
            c, h, w = shape
            b = 1
            x = x.unsqueeze(0)
            squeeze = True
        else:
            assert(False)
        
        assert(c == self.channels)
        out = F.conv2d(x, self.G, padding=self.c, groups=self.channels)
        bo, co, ho, wo = out.size()
        assert(bo == b)
        assert(co == c)
        assert(ho == h)
        assert(wo == w)
        if squeeze:
            out = out.squeeze(0)
        return out




# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet110", "imagenet32_resnet110"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset in ["imagenet", "restricted_imagenet"]:
        model = resnet50(pretrained=False)
        if dataset == "restricted_imagenet":
            model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "imagenet32_resnet110":
        model = resnet_cifar(depth=110, num_classes=1000).cuda()

    # Both layers work fine, We tried both, and they both
    # give very similar results 
    # IF YOU USE ONE OF THESE FOR TRAINING, MAKE SURE
    # TO USE THE SAME WHEN CERTIFYING.
    normalize_layer = get_normalize_layer(dataset)
    # normalize_layer = get_input_center_layer(dataset)
    if dataset == 'cifar10':
        V = VingetteModule((3, 32, 32), 'circ', 2).to('cuda')
    else:
        V = VingetteModule((3, 224, 224), 'circ', 2).to('cuda')
    return torch.nn.Sequential(V, normalize_layer, model)
