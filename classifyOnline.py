from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing, setup_args_getE
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

def sample_transformation(args, img):
    if args.transformation == 'rot':
        angle = np.random.normal(0, args.sigma_gamma, 1)[0]
        img_rot = gt.rotate(img, angle, args.intErr, args.gpu)[0].reshape(img.shape)
    elif args.transformation == 'trans':
        dd = np.random.normal(0, args.sigma_gamma, 2)
        img_rot = gt.translate(img, dd[0], dd[1], args.intErr, args.gpu)[0].reshape(img.shape)
    return img_rot

def pre(args, x):
    if args.resize_post_transform > 0 or args.center_crop_post_transform > 0:
        assert(False)
    return x

parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_ksmooth(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_getE(parser)
parser.add_argument('-E', type=float, default=0.3, help='E')

args = parser.parse_args()
setup(args)
args.gpu = 0 if args.use_cuda else -1
model = get_basemodel(args)
model = setup_model(args, model)
data = get_data(args)
logger = get_logger(args, __file__)

cnt = 0
print("index", "label", "predicted", "label_attack", "E", "rhoE", "gamma", "maxErr", "label_smooth", "radius", "time_analysis", "time_smooth", file=logger)    
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

    pred_clean = run_model(args, model, [img], pre=pre).argmax(dim=1).item()

    for m in range(args.nr_attacks):    
        if pred_clean != label:
            print(idx, label, pred_clean, None, None, None, None, None, None, None, None, None, file=logger)
            continue
        cnt += 1

        #attack with worst of k
        print("Attacking")
        if args.transformation == 'rot':
            params = np.random.uniform(args.gamma0, args.gamma1, args.attack_k)
            imgs_attacked = []
            for i in range(args.attack_k): 
                imgs_attacked.append(gt.rotate(img, params[i], args.intErr, args.gpu)[0].reshape(img.shape))
        elif args.transformation == 'trans':
            params = np.random.uniform(args.gamma0, args.gamma1, (args.attack_k, 2))
            imgs_attacked = []
            for i in range(args.attack_k):
                imgs_attacked.append(gt.translate(img, params[i, 0],
                                                  params[i, 1],
                                                  args.intErr,
                                                  args.gpu
                                                  )[0].reshape(img.shape))
        l = torch.full((args.attack_k,), label, dtype=torch.long, device=args.device)
        logits = run_model(args, model, imgs_attacked)
        with torch.no_grad():
            ce = F.cross_entropy(logits, l, reduce=False)
            n = ce.argmax().cpu().item()
            pred = logits[n, ...].argmax().cpu().item()
        img_a = imgs_attacked[n]
        param_a = params[n, ...]
        print(param_a)
        pred_a = run_model(args, model, img_a).argmax(dim = 1).cpu().item()
        
        print("calculating maximal interpolation error")
        t0 = time.time()
        if args.transformation == 'rot':
            betas = np.random.normal(0, args.sigma_gamma, args.nrBetas)
        elif args.transformation == 'trans':
            betas = np.random.normal(0, args.sigma_gamma, (2, args.nrBetas))
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
                                 refinements=args.refinements,                     
                                 timeout=120)
        errs = np.array(errs)
        E = args.E
        print(errs)
        print(np.sum(errs <= args.E), )
        rhoEalpha = 0.01
        rhoE = (1 - proportion_confint(np.sum(errs <= args.E), args.nrBetas, alpha=2 * rhoEalpha, method="beta")[0]) + rhoEalpha
        t1 = time.time()
        print("max err:", errs.max(), rhoE)
        print("gamma", gamma)
        if errs.max() > args.stop_err:
            print(idx, label, pred_clean, pred_a, E, rhoE, gamma, errs.max(), None, None, t1 - t0, None, file=logger)
            print("error too high")
            continue
        c, R = ksmooth(args,
                       model,
                       img,
                       sample_transformation,
                       [E],
                       [rhoE],
                       pre=pre)
        t2 = time.time()
        print(idx, label, pred_clean, pred_a, E, rhoE, gamma, errs.max(), c, R, t1 - t0, t2 - t1, file=logger)
