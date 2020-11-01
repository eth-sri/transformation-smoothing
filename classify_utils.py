import torch.multiprocessing
from transformations import rotate, translate, Filter, get_vingette_mask
import argparse
import torch
import torch.nn as nn
import sys
from util import str2bool, Logger
import torch.backends.cudnn as cudnn
from datetime import datetime
import random
import os
import glob2 as glob
import numpy as np
from robustness import model_utils, datasets
from robustness.attacker import AttackerModel
import dill
import torchvision.transforms as transforms
import torchvision.models as models
import scipy.stats as sps
import torch.nn.functional as F
from statsmodels.stats.proportion import proportion_confint
from resnet import resnet18
from mnist_net import MNISTConvNet

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        mean = torch.tensor(mean, dtype=torch.float)
        assert(len(mean.size()) == 1)
        assert(mean.size(0) == 3)
        mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.mean = nn.Parameter(mean, requires_grad=False)

        std = torch.tensor(std, dtype=torch.float)
        assert(len(std.size()) == 1)
        assert(std.size(0) == 3)
        std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.std = nn.Parameter(std, requires_grad=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        return (x - self.mean.repeat(b, 1, h, w)) / self.std.repeat(b, 1, h, w)

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


class RobustnessModelInnerWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(x)


def setup_args(parser):
    parser.add_argument('--model', type=str, required=True, help='type of network')
    parser.add_argument('--seed', type=int, default='1', help='seed')
    parser.add_argument('-N', type=int, default='1', help='number of samples')
    parser.add_argument('--Nstart', type=int, default='0', help='first index of samples to consider')
    parser.add_argument('--Ncnt', type=int, default='0', help='subset of the sample to consider; 0 considers all')
    parser.add_argument('--dataset', choices=['imagenet',
                                              'restricted_imagenet',
                                              'mnist',
                                              'GTSRB',
                                              'cifar',
                                              'fashionmnist'],
                        default='imagenet', help='dataset')
    parser.add_argument('--debug', type=str2bool, default=False,
                    help='enable additional debug output; mostly for the C++ backend')
    parser.add_argument('--gpu', type=str2bool, default=True,
                    help='use gpu')    
    parser.add_argument('--attack-k', type=int, default=10, help='number of attacks for worst-of-k')
    parser.add_argument('--nr-attacks', type=int, default=3, help='')
    parser.add_argument('--gamma', type=float, default=10, help='attacker parameter in [-gamma, gamma]')
    parser.add_argument('--sigma-gamma', type=float, default=10, help='sigma used to smooth over gamma')
    parser.add_argument('--alpha-gamma', type=float, default=0.01, help='alpha for the smoothing over gamma')
    parser.add_argument('--n0-gamma', type=int, default=10, help='n0 (size of the initial sample to determine target class) for the smoothing over gamma')
    parser.add_argument('--n-gamma', type=int, default=10, help='number of samples for the smoothing estimate over gamma')
    parser.add_argument('--name', type=str, default=None, help='name of the experiment')
    parser.add_argument('--intErr', type=str2bool, default=True, help='also consider integer error in interpolation')
    parser.add_argument('--transformation', choices=['rot', 'trans'], default='rot', help='transformation to consider')
    return parser

def setup_args_preprocessing(parser):
    parser.add_argument('--resize', type=int, default=0, help='')
    parser.add_argument('--radiusDecrease', type=float, default=-1, help='')
    parser.add_argument('--resize-post-transform', type=int, default=0, help='')
    parser.add_argument('--center-crop-post-transform', type=int, default=0, help='')
    parser.add_argument('--filter-sigma', type=float, default=0, help='')
    parser.add_argument('--filter-size', type=int, default=5, help='')
    return parser

def setup_args_getE(parser):
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use in error estimation')
    parser.add_argument('--gt-batch-size', type=int, default=20, help='batch size for error estimation')
    parser.add_argument('--target-err', type=float, default=0.3, help='')
    parser.add_argument('--stop-err', type=float, default=2.0, help='')
    parser.add_argument('--initial-splits', type=int, default=100, help='')
    parser.add_argument('--refinements', type=int, default=10, help='')
    parser.add_argument('--nrBetas', type=int, default=100, help='')
    parser.add_argument('--nrBetasSplit', type=int, default=500, help='')
    return parser

def get_basemodel(args):
    if args.model == 'none': return None

    if args.model == 'resnet50' and args.dataset == 'imagenet':
        model = models.resnet50(pretrained=True).eval()
        normalize = NormalizeLayer(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    else:
        path = glob.glob(os.path.join('./models', args.model, '**', 'checkpoint.pt.best'))
        path_tar = glob.glob(os.path.join('./models', args.model, '**', 'checkpoint.pth.tar'))
        if not (len(path) > 0 or (len(path_tar) > 0 and args.dataset in ['cifar', 'imagenet']) ):
            print("Could not load model")
            sys.exit(1)

        if len(path_tar) > 0 and args.dataset in ['cifar', 'imagenet', 'restricted_imagenet']:
            sys.path.append('smoothing-adversarial/code')
            from architectures import get_architecture
            path = path_tar[0]
            print('Loading model from', path)
            checkpoint = torch.load(path, map_location='cpu')
            if args.dataset == 'cifar':
                model = get_architecture(checkpoint["arch"], 'cifar10')
            else:
                model = get_architecture(checkpoint["arch"], args.dataset)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to('cpu')
            for i, m in enumerate(model):
                if isinstance(m, torch.nn.DataParallel):
                    model[i] = m.module
            normalize = None
            model = model[1:]
            print(model)
        else:
            path = path[0]
            print('Loading model from', path)
            if args.dataset in ['imagenet', 'restricted_imagenet', 'cifar']:
                ds_class = datasets.DATASETS[args.dataset]
                ds = ds_class("./ds/imagenet" if args.dataset != 'cifar' else './ds/cifar')
                model, _ = model_utils.make_and_restore_model(arch=('resnet18' if args.dataset == 'cifar' else 'resnet50'),
                                                            dataset=ds,
                                                            resume_path=path,
                                                            parallel=False)
                normalize = model.normalizer
                model = model.model
            elif args.dataset in ['mnist', 'fashionmnist', 'GTSRB']:
                if 'mnist' in args.dataset:
                    num_classes = 10
                    color_channels = 1
                    mean = torch.tensor([0.1307])
                    std = torch.tensor([0.3081])
                    if 'convnet' in path:
                        print('convenet')
                        model = MNISTConvNet()
                    else:
                        model = resnet18(num_classes=num_classes, color_channels=color_channels)
                elif args.dataset == 'GTSRB':
                    num_classes = 43
                    color_channels = 3
                    mean = torch.tensor([0.3337, 0.3064, 0.3171])
                    std = torch.tensor([0.2672, 0.2564, 0.2629])
                    model = resnet18(num_classes=num_classes, color_channels=color_channels)
                model = RobustnessModelInnerWrapper(model)
                d = argparse.Namespace()
                d.mean = mean
                d.std = std
                model = AttackerModel(model, d)
                checkpoint = torch.load(path, pickle_module=dill)
                state_dict_path = 'model'
                if not ('model' in checkpoint):
                    state_dict_path = 'state_dict'
                sd = checkpoint[state_dict_path]
                sd = {k[len('module.'):]:v for k, v in sd.items()}
                sd = {(k if 'model.net' in k else k.replace('model.', 'model.net.')):v for k, v in sd.items()}
                model.load_state_dict(sd)
                normalize = model.normalizer
                model = model.model
            else:
                assert(False)
    m = []
    if normalize is not None:
        m.append(normalize.to(args.device)) 
    if args.radiusDecrease >= 0:
        shape={'rot': 'circ', 'trans':'rect'}[args.transformation]
        size = {'mnist': (1, 28, 28),
                'fashionmnist': (1, 28, 28),
                'cifar': (3, 32, 32),
                'GTSRB': (3, np.inf, np.inf),
                'imagenet': (3, np.inf, np.inf),
                'restricted_imagenet': (3, np.inf, np.inf)}[args.dataset]
        if args.resize_post_transform > 0:
            size = (size[0],
                    min(size[1], args.resize_post_transform),
                    min(size[2], args.resize_post_transform))
        if args.center_crop_post_transform > 0:
            size = (size[0],
                    min(size[1], args.center_crop_post_transform),
                    min(size[2], args.center_crop_post_transform))    
        V = VingetteModule(size, shape, args.radiusDecrease)
        m.append(V)
    m.append(model)
    model = torch.nn.Sequential(*m)
    if args.use_cuda:
        model = torch.nn.DataParallel(model.to(args.device))
    model = model.eval().to(args.device)
    return model


def setup(args):
    use_cuda = torch.cuda.is_available() and args.gpu
    args.device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        # ensure data parallel works
        torch.cuda.set_device(0)
    args.use_cuda = use_cuda
    args.ps_debug = args.debug
    cudnn.bechmark = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.gamma0 = -args.gamma
    args.gamma1 = args.gamma

def get_logger(args, fn):
    if args.name is not None:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        name = f"{args.name}_{current_time}.txt"
        prefix = os.path.join('results', os.path.basename(fn).replace('.py', ''))
        os.makedirs(prefix, exist_ok=True)
        fn = os.path.join(prefix, name)
        print(fn)
        logger = Logger(fn, sys.stdout)
    else:
        logger = sys.stdout
    return logger
    
def get_data(args, split='val'):
    from data import get_dataset
    ds = get_dataset(args, split)
    I = list(range(len(ds)))
    n = min(args.N, len(I))
    samples = random.sample(I, n)
    Ncnt = args.Ncnt if args.Ncnt > 0 else (args.N - args.Nstart)
    samples = samples[args.Nstart:(args.Nstart+Ncnt)]
    data = [ds[i] for i in samples]
    return data
