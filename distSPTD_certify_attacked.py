from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing
from classify_ksmooth import ksmooth, setup_args_ksmooth, run_model, sample_transformation, pre
import argparse
import torch
import numpy as np
from data import get_dataset
import geometrictools as gt
from collections import Counter
import torch.nn as nn
import time
from util import str2bool
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from statsmodels.stats.proportion import proportion_confint
from transformations import rotate, translate
import PIL
import multiprocessing as mp

def rotate_(data):
    img, angle = data
    return TF.rotate(img, angle, resample=PIL.Image.BILINEAR)

def translate_(data):
    img, dx, dy = data
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=PIL.Image.BILINEAR)

def getRotations(img, angles):
    angles = angles.tolist()
    images = [(img, angle) for angle in angles]
    with mp.Pool() as p:
        images = p.map(rotate_, images)
    return images

def getTranslations(img, offset):
    assert(len(offset.shape) == 2)
    assert(offset.shape[0] == 2)
    dx = offset[0, ...]
    dy = offset[0, ...]
    n = offset.shape[1]
    images = [(img, dx[i], dy[i]) for i in range(n)]
    with mp.Pool() as p:
        images = p.map(translate_, images)
    return images


def attack(args, img):
    if args.transformation == 'rot':
        params = np.random.uniform(args.gamma0, args.gamma1, args.attack_k)
        images = getRotations(img, params)
    elif args.transformation == 'trans':
        params = np.random.uniform(args.gamma0, args.gamma1, (args.attack_k, 2))
        images = getTranslations(img, params)
    l = torch.full((args.attack_k,), label, dtype=torch.long, device=args.device)
    logits = run_model(args, model, images)
    with torch.no_grad():
        ce = F.cross_entropy(logits, l, reduce=False)
        n = ce.argmax().cpu().item()
        pred = logits[n, ...].argmax().cpu().item()
    img_a = images[n]
    param_a = params[n, ...]
    #print(param_a)
    return img_a, param_a

parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_ksmooth(parser)

args = parser.parse_args()
setup(args)
args.gpu = 0 if args.use_cuda else -1
model = get_basemodel(args)
data = get_data(args)
logger = get_logger(args, __file__)

print(args, file=logger)
args.interpolation = {'nearest': PIL.Image.NEAREST,
                      'bilinear': PIL.Image.BILINEAR,
                      'bicubic': PIL.Image.BICUBIC}[args.interpolation]

print("index", "label", "predicted", "predicted_attacked", "label_smooth", "radius", "time", file=logger)

for idx, d in enumerate(data):
    print()
    img, label = d

    if args.resize > 0:
        img = TF.resize(img, args.resize)
        
    pred_clean = run_model(args, model, [img], pre=pre).argmax(dim=1).item()
    for m in range(args.nr_attacks):    
        #attack with worst of k
        print("Attacking")
        img_a, _ = attack(args, img)

        pred_a = run_model(args, model, [img_a], pre=pre).argmax(dim=1).item()
        t0 = time.time()
        c, R = ksmooth(args,
                        model,
                        img_a,
                        sample_transformation,
                        args.E,
                        args.rhoE,
                        pre=pre)
        t1 = time.time()
    print(idx+args.Nstart, label, pred_clean, pred_a, c, R, t1 - t0, file=logger, flush=True)
