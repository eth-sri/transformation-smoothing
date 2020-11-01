from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing
import torch
from statsmodels.stats.proportion import proportion_confint
import scipy.stats as sps
from collections import Counter
import torch.nn as nn
from tqdm import trange
import time
from util import str2bool
import torchvision.transforms.functional as TF
import numpy as np

def _smooth(args,
            model,
            img,
            Es,
            rhoEs,
            masks,
            ns,
            alphas,
            sigmas,
            C0):
    assert(len(ns) == len(masks))
    assert(len(ns) == len(alphas))
    assert(len(ns) == len(sigmas))
    assert(len(ns) == len(Es) + 1)
    assert(len(Es) == len(rhoEs))
    
    mask, masks = masks[0], masks[1:]
    alpha, alphas = alphas[0], alphas[1:]
    sigma, alphas = sigmas[0], sigmas[1:]
    n, ns = ns[0], ns[1:]
    E, Es = (Es[0], Es[1:]) if len(Es) > 0 else None, None
    rhoE, rhoEs = (rhoEs[0], rhoEs[1:]) if len(rhoEs) > 0 else None, None
    
    def _sample():
        cs = []
        channels = {'F': list(range(img.shape[1])),
                    'R': [0],
                    'G': [1],
                    'B': [2]}[mask]
        M = torch.zeros_like(img)
        for c in channels:
            M[:, c, :, :] = 1
        if E is None: # inner_most classifier
            m = n
            while m > 0:
                batch_size = min(args.batch_size, m)        
                batch = img.repeat(batch_size, 1, 1, 1)
                noise = sigma * M.repeat(batch_size, 1, 1, 1) * torch.randn_like(batch)
                labels = model(batch).argmax(dim=1).detach().cpu().numpy().tolist()
                cs.extend(labels)
                m -= batch_size
        else:
            for i in range(n):
                c, r = _smooth(args,
                               model,
                               img + M * torch.randn_like(img),
                               Es,
                               rhoEs,
                               masks,
                               ns,
                               alphas,
                               sigmas,
                               C0)
                if r >= E:
                    cs.append(c)
        return Counter(cs)

    sample = _sample()
    cnt = sample[C0] if C0 in sample else 0
    p = proportion_confint(cnt,
                           n,
                           alpha=2*alpha,
                           method="beta")[0]
    if E is not None:
        p = p - (alphas[0]) # alpha corresponding to the 1 deeper classifier
    R = sps.norm.ppf(p)
    R = R * sigma if p >= 0.5 else 0
    return C0, R

def ksmooth(args,
            model,
            img,
            sample_transformation,
            Es,
            rhoEs,
            pre = None):
    assert(len(Es) == len(rhoEs))
    E, Es = Es[0], Es[1:]
    rhoE, rhoEs = rhoEs[0], rhoEs[1:]

    C0_sample = [sample_transformation(args, img) for i in range(args.batch_size)]
    if pre is not None:
        C0_sample = [pre(args, img) for img in C0_sample]
    C0_sample = [TF.to_tensor(img) if not isinstance(img, torch.Tensor) else img for img in C0_sample]
    C0_sample = torch.stack(C0_sample).to(args.device)
    C0_sample = model(C0_sample).argmax(dim=1).cpu().numpy().tolist()
    C0 = Counter(C0_sample).most_common(1)[0][0]
    cnt = 0
    for i in trange(args.n_gamma):
        img_s = sample_transformation(args, img)
        if pre is not None:
            img_s = pre(args, img_s)
        if not isinstance(img_s, torch.Tensor):
            img_s = TF.to_tensor(img_s)
        img_s = img_s.unsqueeze(0).to(args.device)
        _, r = _smooth(args,
                       model,
                       img_s,
                       Es,
                       rhoEs,
                       args.maskI,
                       args.nI,
                       args.alphaI,
                       args.sigmaI,
                       C0)        
        if (r >= E):
            cnt += 1
    pouter = proportion_confint(cnt,
                                args.n_gamma,
                                alpha=2*args.alpha_gamma,
                                method="beta")[0] - (args.alphaI[0] + rhoE)
    R_outer = sps.norm.ppf(pouter)
    R_outer = R_outer * args.sigma_gamma if pouter >= 0.5 else 0
    print(C0, cnt, rhoE, pouter, R_outer)
    return C0, R_outer

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
    if not isinstance(imgs, list):
        imgs = [imgs]
    if pre is not None:
        imgs = [pre(args, img) for img in imgs]
    imgs = [TF.to_tensor(img) if not isinstance(img, torch.Tensor) else img for img in imgs]
    imgs = torch.stack(imgs).to(args.device)
    with torch.no_grad():
        out = model(imgs)
    return out

def setup_args_ksmooth(parser):
    parser.add_argument('--sigmaI', type=float, default=[0.3], help='sigmaR', nargs='+')
    parser.add_argument('--alphaI', type=float, default=[0.001], help='', nargs='+')
    parser.add_argument('-nI', type=int, default=[200], help='n', nargs='+')
    parser.add_argument('--maskI', choices=['F', 'R', 'G', 'B'], default=['F'], help='n', nargs='+')
    parser.add_argument('--batch-size', type=int, default=2000, help='')
    return parser


def setup_model(args, model):
    from LowPassLayer import Filter
    m = []
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

    if args.filter_sigma > 0:
        m.append(Filter(args.filter_size,
                        args.filter_sigma,
                        1 if args.dataset == 'mnist' else 3).to(args.device))
    m.append(model)
    model = nn.Sequential(*m).to(args.device)
    return model
