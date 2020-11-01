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

print("index", "label", "predicted", "label_smooth", "radius", "time", file=logger)

for idx, d in enumerate(data):
    print()
    img, label = d

    if args.resize > 0:
        img = TF.resize(img, args.resize)

    pred_clean = run_model(args, model, [img], pre=pre).argmax(dim=1).item()
    t0 = time.time()
    c, R = ksmooth(args,
                    model,
                    img,
                    sample_transformation,
                    args.E,
                    args.rhoE,
                    pre=pre)
    t1 = time.time()
    print(idx+args.Nstart, label, pred_clean, c, R, t1 - t0, file=logger, flush=True)
