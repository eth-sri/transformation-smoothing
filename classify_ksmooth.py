from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing, VingetteModule
import torch
from statsmodels.stats.proportion import proportion_confint
import scipy.stats as sps
from collections import Counter
import torch.nn as nn
from tqdm import trange
import time
from util import str2bool
import torchvision.transforms.functional as TF
from transformations import rotate, translate, Filter, get_vingette_mask
import numpy as np


def l2_sample(args,
            model,
            img,
            n,
            sigma):
    cs = []
    m = n
    while m > 0:
        batch_size = min(args.batch_size, m)        
        batch = img.repeat(batch_size, 1, 1, 1)
        noise = sigma * torch.randn_like(batch)
        labels = model(batch + noise).argmax(dim=1).detach().cpu().numpy().tolist()
        cs.extend(labels)
        m -= batch_size
    return Counter(cs)


def l2_smooth(args,
            model,
            img,
            n,
            alpha,
            sigma,
            C0):
    sample = l2_sample(args, model, img, n, sigma)
    cnt = sample[C0] if C0 in sample else 0
    p = proportion_confint(cnt,
                           n,
                           alpha=2*alpha,
                           method="beta")[0]
    R = sps.norm.ppf(p)
    R = R * sigma if p >= 0.5 else None 
    return C0, R

def ksmooth(args,
            model,
            img,
            sample_transformation,
            E,
            rhoE,
            pre = None):
    if model == None: return None, None

    fl = []
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
        V = VingetteModule(size, shape, args.radiusDecrease).to(args.device)
        fl.append(V)
    if args.filter_sigma > 0:
        fl.append(Filter(args.filter_size,
                    args.filter_sigma,
                    1 if args.dataset == 'mnist' else 3).to(args.device))
    fl = nn.Sequential(*fl)

    # determine guess for top class
    C0_sample = [sample_transformation(args, img) for i in range(args.n0_gamma)]
    if pre is not None:
        C0_sample = [pre(args, img) for img in C0_sample]
    C0_sample = [TF.to_tensor(img) if not isinstance(img, torch.Tensor) else img for img in C0_sample]
    C0_out = []
    k = 0
    while len(C0_out) < args.n0_gamma:
        sample = torch.stack(C0_sample[k*args.batch_size:min((k+1)*args.batch_size, args.n0_gamma)]).to(args.device)
        noise = args.sigma_eps * torch.randn_like(sample)
        C0_out.extend(model(fl(sample) + noise).argmax(dim=1).cpu().numpy().tolist())
    C0 = Counter(C0_out).most_common(1)[0][0]

    cnt = 0
    for i in trange(args.n_gamma):
        img_s = sample_transformation(args, img)
        if pre is not None:
            img_s = pre(args, img_s)
        if not isinstance(img_s, torch.Tensor):
            img_s = TF.to_tensor(img_s)
        img_s = img_s.unsqueeze(0).to(args.device)
        _, r = l2_smooth(args,
                       model,
                       fl(img_s),
                       args.n_eps,
                       args.alpha_eps / args.n_gamma,
                       args.sigma_eps,
                       C0)        
        if r is not None and r >= E:
            cnt += 1
    pouter = proportion_confint(cnt,
                                args.n_gamma,
                                alpha=2*args.alpha_gamma,
                                method="beta")[0] - rhoE
    #print('p', pouter)
    R_outer = sps.norm.ppf(pouter)
    R_outer = R_outer * args.sigma_gamma if pouter >= 0.5 else None
    #print(C0, cnt, rhoE, pouter, R_outer)
    return C0, R_outer

def test_predict(cnts, alpha):
    cnts = sample.most_common(2)
    ca, na = cnts[0]
    if len(cnts) > 1:
        _, nb = cnts[1]
    else:
        nb = n - na
    if sps.binom_test(na, na + nb, p=0.5) > alpha:
        return -1
    else:
        return ca

def l2_predict(args,
            model,
            img,
            n,
            alpha,
            sigma):
    sample = l2_sample(args, model, n, sigma)
    return test_predict(sample)


def kpredict(args,
            model,
            img,
            sample_transformation,
            pre = None):
    if model == None: return None, None

    
    fl = []
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
        V = VingetteModule(size, shape, args.radiusDecrease).to(args.device)
        fl.append(V)
    if args.filter_sigma > 0:
        fl.append(Filter(args.filter_size,
                    args.filter_sigma,
                    1 if args.dataset == 'mnist' else 3).to(args.device))
    fl = nn.Sequential(*fl)

    cnts = [] 
    for i in trange(args.n_gamma):
        img_s = sample_transformation(args, img)
        if pre is not None:
            img_s = pre(args, img_s)
        if not isinstance(img_s, torch.Tensor):
            img_s = TF.to_tensor(img_s)
        img_s = img_s.unsqueeze(0).to(args.device)
        cnts.append(l2_predict(args, model, fl(img_s), n, args.alphaI/args.n_gamma, sigma))
    cnts = Counter(cnts)
    return test_predict(cnts)


class VingetteModule(nn.Module):
    def __init__(self, size, shape, offset):
        super().__init__()
        from transformations import get_vingette_mask
        V = get_vingette_mask(size,
                              shape_type=shape,
                              offset=offset)
        V = torch.tensor(V, dtype=torch.float).unsqueeze(0)
        self.V = nn.Parameter(V, requires_grad=False)
    def forward(self, x):
        return self.V * x


def run_model(args, model, imgs, pre=None):
    if model is None: return None
    if not isinstance(imgs, torch.Tensor):
        if not isinstance(imgs, list):
            imgs = [imgs]
        if pre is not None:
            imgs = [pre(args, img) for img in imgs]
        imgs = [TF.to_tensor(img) if not isinstance(img, torch.Tensor) else img for img in imgs]
        imgs = torch.stack(imgs).to(args.device)
    else:
        imgs = imgs.to(args.device)
    if args.filter_sigma > 0:
        fl = Filter(args.filter_size,
                    args.filter_sigma,
                    1 if args.dataset == 'mnist' else 3).to(args.device)
        imgs = fl(imgs)

    with torch.no_grad():
        out = model(imgs)
    return out

def setup_args_ksmooth(parser):
    parser.add_argument('--sigma-eps', type=float, default=0.3)
    parser.add_argument('--alpha-eps', type=float, default=0.001, help='alpha for all the l2 tests, will divided by n-gamma')
    parser.add_argument('--n-eps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2000, help='')
    parser.add_argument('-E', type=float, default=0.90)
    parser.add_argument('--rhoE', type=float, default=0.01)
    parser.add_argument('--interpolation', choices=['nearest', 'bilinear', 'bicubic'],
                        default='bilinear', help='interpolation method')
    return parser

def sample_transformation(args, img):
    if args.transformation == 'rot':
        angle = np.random.normal(0, args.sigma_gamma, 1)[0]
        img_T = rotate(img, angle, resample=args.interpolation)
    elif args.transformation == 'trans':
        dd = np.random.normal(0, args.sigma_gamma, 2)
        img_T = translate(img, dd, resample=args.interpolation)
    return img_T

def pre(args, x):
    if args.resize_post_transform > 0:
        x = TF.resize(x, args.resize_post_transform)
    if args.center_crop_post_transform > 0:
        x = TF.center_crop(x, args.center_crop_post_transform)
    return x



