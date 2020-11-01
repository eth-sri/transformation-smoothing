from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing
from util import Logger
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import scipy.stats as sps
from collections import Counter
import sys
from datetime import datetime
import time
import torchvision.transforms.functional as TF
import PIL, PIL.Image
import multiprocessing as mp
import os

def pre(args, x):
    if args.dataset == 'imagenet':
        x = TF.resize(x, 256)
        x = TF.center_crop(x, 224)
    elif args.dataset == 'GTSRB':
        x = TF.resize(x, 32)
        x = TF.center_crop(x, 32)
    return x

def rotate(data):
    img, angle = data
    return TF.rotate(img, angle, resample=PIL.Image.BILINEAR)

def translate(data):
    img, dx, dy = data
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=PIL.Image.BILINEAR)

def getRotations(img, angles):
    angles = angles.tolist()
    images = [(img, angle) for angle in angles]
    with mp.Pool() as p:
        images = p.map(rotate, images)
    return images

def getTranslations(img, offset):
    assert(len(offset.shape) == 2)
    assert(offset.shape[0] == 2)
    dx = offset[0, ...]
    dy = offset[0, ...]
    n = offset.shape[1]
    images = [(img, dx[i], dy[i]) for i in range(n)]
    with mp.Pool() as p:
        images = p.map(translate, images)
    return images

def worst_of_k(args, model, img, label):
    if args.transformation == 'rot':
        angles = np.random.uniform(args.gamma0, args.gamma1, args.attack_k)
        images = getRotations(img, angles)
    elif args.transformation == 'trans':
        offset = np.random.uniform(args.gamma0, args.gamma1, (2, args.attack_k))
        images = getTranslations(img, offset)
    data = [TF.to_tensor(pre(args, img)) for img in images]
    data = torch.stack(data).to(args.device)
    l = torch.full((args.attack_k,), label, dtype=torch.long, device=args.device)    
    logits = model(data)
    with torch.no_grad():
        ce = F.cross_entropy(logits, l, reduce=False)
        n = ce.argmax().cpu().item()
        pred = logits[n, ...].argmax().cpu().item()
    return images[n], pred

def classify(args, model, img):
    img_pt = TF.to_tensor(pre(args, img)).unsqueeze(0).to(args.device)    
    with torch.no_grad():
        pred = model(img_pt).argmax(dim=1).cpu().item()
    return pred

def sample(args, model, img, n, delta=None):
    if args.transformation == 'rot':
        if delta is None:
            delta = 0
        else:
            assert (isinstance(delta, float) or
                    (isinstance(delta, np.ndaray) and len(delta.shape) == 1) and delta.shape[0] == 1)
        angles = np.random.normal(scale=args.sigma_gamma, size=n) + delta
        images = getRotations(img, angles)
    elif args.transformation == 'trans':
        if delta is None:
            delta = 0
        else:
            print(delta)
            assert isinstance(delta, np.ndarray) and len(delta.shape) == 2 and delta.shape[0] == 2 and delta.shape[1] == 1
        offset = np.random.normal(scale=args.sigma_gamma, size=(2,n)) + delta
        images = getTranslations(img, offset)
    data = [TF.to_tensor(pre(args, img)) for img in images]
    data = torch.stack(data).to(args.device)
    with torch.no_grad():
        pred = model(data).argmax(dim=1).cpu().numpy().tolist()
    return Counter(pred)

def smooth(args, model, img):
    C0, _ = sample(args, model, img, args.n0_gamma).most_common(1)[0]
    counts = sample(args, model, img, args.n_gamma)
    C0count = counts[C0] if C0 in counts else 0
    p = proportion_confint(C0count, args.n_gamma, alpha=2 * args.alpha_gamma, method="beta")[0]
    R = sps.norm.ppf(p)
    R = R * args.sigma_gamma if p >= 0.5 else 0
    return C0, R


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = setup_args(parser)
    parser = setup_args_preprocessing(parser)
    args = parser.parse_args()
    setup(args)
    model = get_basemodel(args)
    data = get_data(args)
    logger = get_logger(args, __file__)
    print(args, file=logger)
    print("index", "label", "predicted", "label_attack", "label_smooth", "radius", "time", file=logger)    
    for i, d in enumerate(data):
        img, label = d
        pred = classify(args, model, img)
        img_a, label_a = worst_of_k(args, model, img, label)
        t0 = time.time()
        pred_s, R = smooth(args, model, img)
        t1 = time.time()
        print(i, label, pred, label_a, pred_s, R, t1 - t0, file=logger)
