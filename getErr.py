from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing, setup_args_getE
from util import Logger
import argparse
import torch
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
import geometrictools as gt



parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_getE(parser)
parser.add_argument('--betaSplit', type=int, default=-1, help='split beta in chunks for error bounding; higher numbers make the error computation faster; lower numbers more precise; -1 diables it and considers all errors together')
args = parser.parse_args()
setup(args)
model = get_basemodel(args)
data = get_data(args)
logger = get_logger(args, __file__)
print(args, file=logger)

print("index", "err", file=logger, flush=True)
for idx, d in enumerate(data):
    print()
    img, label = d

    if args.resize > 0:
        img = TF.resize(img, args.resize)
        img = TF.center_crop(img, args.resize)
        
    img = np.array(img)   
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    
    assert(img.dtype == np.uint8)
    img = img.astype(dtype=np.float32)/255.0
    
    if args.transformation == 'rot':
        betas = np.random.normal(0, args.sigma_gamma, (1, args.nrBetas))
    elif args.transformation == 'trans':
        betas = np.random.normal(0, args.sigma_gamma, (2, args.nrBetas))
    k = 0
    errs, gamma = [], np.array([0] * betas.shape[0])
    s = args.betaSplit if args.betaSplit > 0 else args.nrBetas
    while len(errs) < args.nrBetas:
        errs_, gamma_ = gt.getE(image=img,
                                transformation=args.transformation,
                                targetE=args.target_err,
                                stopE=args.stop_err,
                                gamma0=args.gamma0,
                                gamma1=args.gamma1,
                                betas=betas[:, k*s:(k+1)*s],
                                invertT=False,
                                resizePostTransform=args.resize_post_transform,
                                centerCropPostTranform=args.center_crop_post_transform,
                                filterSigma=args.filter_sigma,
                                filterSize=args.filter_size,
                                radiusDecrease=args.radiusDecrease,
                                initialSplits=args.initial_splits,
                                batchSize=args.gt_batch_size,
                                threads=args.threads,
                                gpu=args.use_cuda,
                                debug=args.debug,
                                doInt=args.intErr,
                                refinements=args.refinements,                                              
                                timeout=120)
        errs.extend(errs_)
    for err in errs:
        print(idx+args.Nstart, err, file=logger, flush=True)
