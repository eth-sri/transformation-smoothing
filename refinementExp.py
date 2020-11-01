from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing, setup_args_getE
import argparse
import torch
import numpy as np
from data import get_dataset
import geometrictools as gt
from collections import Counter
import time
from util import str2bool
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_getE(parser)
args = parser.parse_args()
setup(args)
args.gpu = 0 if args.use_cuda else -1
data = get_data(args)
logger = get_logger(args, __file__)

for idx, d in enumerate(data):
    print()
    img, label = d

    if args.resize > 0:
        img = TF.resize(img, args.resize)
        
    img = np.array(img)   
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    
    assert(img.dtype == np.uint8)
    img = img.astype(dtype=np.float32)/255.0

    for m in range(args.nr_attacks):    
        if args.transformation == 'rot':
            betas = np.random.normal(0, args.sigma_gamma, args.nrBetas)
            params = np.random.uniform(args.gamma0, args.gamma1, 1)
            img_a = gt.rotate(img, params[0], args.intErr, args.gpu)[0].reshape(img.shape)
        elif args.transformation == 'trans':
            betas = np.random.normal(0, args.sigma_gamma, (args.nrBetas, 2))
            params = np.random.uniform(args.gamma0, args.gamma1, (1, 2))
            img_a = gt.translate(img, params[k, 0],
                                 params[k, 1],
                                 args.intErr,
                                 args.gpu)[0].reshape(img.shape)
        for refinements in [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]:
            t0 = time.time()
            errs, gamma = gt.getE(image=img_a,
                                     transformation=args.transformation,
                                     targetE=args.target_err,
                                     stopE=args.stop_err,
                                     gamma0=args.gamma0,
                                     gamma1=args.gamma1,
                                     betas=betas,
                                     invertT=True,
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
                                     refinements=refinements,
                                     timeout=120)
            t1 = time.time()
            t = t1 - t0
            errs = np.array(errs)
            err = errs.max()
            print(idx, m, refinements, err, t, file=logger)
