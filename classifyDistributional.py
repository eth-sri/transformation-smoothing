from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing
from classify_ksmooth import ksmooth, setup_args_ksmooth, run_model, setup_model
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

parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_ksmooth(parser)
parser.add_argument('-E', type=float, default=[0.90], help='E', nargs='+')
parser.add_argument('--rhoE', type=float, default=[0.01], help='E', nargs='+')
parser.add_argument('--interpolation', choices=['nearest', 'bilinear', 'bicubic'],
                    default='bilinear', help='interpolation method')

args = parser.parse_args()
setup(args)
args.gpu = 0 if args.use_cuda else -1
basemodel = get_basemodel(args)
model = setup_model(args, basemodel)
data = get_data(args)
logger = get_logger(args, __file__)

print(args, file=logger)
args.interpolation = {'nearest': PIL.Image.NEAREST,
                      'bilinear': PIL.Image.BILINEAR,
                      'bicubic': PIL.Image.BICUBIC}[args.interpolation]

print("index", "label", "predicted", "label_smooth", "radius", "time", file=logger)

for idx, d in enumerate(data):
    print()
    img, label = d

    if args.resize > 0:
        img = TF.resize(img, args.resize)

    pred_clean = run_model(args, basemodel, [img], pre=pre).argmax(dim=1).item()
    if pred_clean != label:
        print(idx, label, pred_clean, None, None, None, file=logger)
    else:
        t0 = time.time()
        c, R = ksmooth(args,
                       model,
                       img,
                       sample_transformation,
                       args.E,
                       args.rhoE,
                       pre=pre)
        t1 = time.time()
        print(idx, label, pred_clean, c, R, t1 - t0, file=logger)
