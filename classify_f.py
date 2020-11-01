from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing
from classify_ksmooth import ksmooth, setup_args_ksmooth, run_model, pre
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
import torchvision.transforms as transforms
import PIL, PIL.Image
import multiprocessing as mp
import os
from data import get_dataset
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = setup_args(parser)
    parser = setup_args_ksmooth(parser)
    parser = setup_args_preprocessing(parser)
    args = parser.parse_args()
    setup(args)
    model = get_basemodel(args)
    ts = []
    if args.resize_post_transform > 0:
        ts.append(transforms.Resize(args.resize_post_transform))
    if args.center_crop_post_transform > 0:
        ts.append(transforms.CenterCrop(args.center_crop_post_transform))
    ts.append(transforms.ToTensor())
    ds = get_dataset(args, 'val', transform=transforms.Compose(ts))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    cnt, corr = 0, 0  
    for x, y in tqdm(loader):
        pred = run_model(args, model, x).argmax(dim=1).cpu()
        corr += (pred == y).sum().item()
        cnt += y.shape[0]
    print(corr/cnt)
