from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing, setup_args_getE
from classify_ksmooth import ksmooth, setup_args_ksmooth, run_model 
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


def attack(args, img):
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
    #print(param_a)
    return img_a, param_a


parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_ksmooth(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_getE(parser)
parser.add_argument('--E', type=float, default=None, help='')
parser.add_argument('--Efile', type=str, default=None, help='load E and rho from file')
parser.add_argument('--betaSplit', type=int, default=-1, help='split beta in chunks for error bounding; higher numbers make the error computation faster; lower numbers more precise; -1 diables it and considers all errors together')
parser.add_argument('--alpha-E', type=float, default=0.001, help='confidence for E\rhoE estimate')

args = parser.parse_args()
setup(args)

if args.Efile is not None:
    df = pd.read_csv(args.Efile, skiprows=2, sep=' ', names=["index", "label", "predicted", "label_attack", "E", "rhoE", "gamma", "maxErr", "label_smooth", "radius", "time_analysis", "time_smooth"])
    e_rho = np.array(df[['E', 'rho', 'maxErr', 'gamma']])
    n = e_rho.shape[0]
    def gen():
        for i in range(n):
            e, rho, me, gamma = e_rho[i, 0], e_rho[i, 1], e_rho[i, 2], e_rho[i, 3]
            yield (e, rho, me, gamma)
    args.Efile = gen()
    for i in range(args.Nstart): next(args.Efile)


args.gpu = 0 if args.use_cuda else -1
model = get_basemodel(args)
data = get_data(args)
logger = get_logger(args, __file__)

print(args, file=logger)
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
        cnt += 1

        #attack with worst of k
        print("Attacking")
        img_a, param_a = attack(args, img)
        pred_a = run_model(args, model, img_a).argmax(dim = 1).cpu().item()
        
        print("calculating maximal interpolation error")
        t0 = time.time()
        if args.transformation == 'rot':
            betas = np.random.normal(0, args.sigma_gamma, (1, args.nrBetas))
        elif args.transformation == 'trans':
            betas = np.random.normal(0, args.sigma_gamma, (2, args.nrBetas))
        if args.Efile is None:
            errs, gamma = [], np.array([0] * betas.shape[0])
            s = args.betaSplit if args.betaSplit > 0 else args.nrBetas
            k = 0
            while len(errs) < args.nrBetas:
                errs_, gamma_ = gt.getE(image=img_a,
                                        transformation=args.transformation,
                                        targetE=args.target_err,
                                        stopE=args.stop_err,
                                        gamma0=args.gamma0,
                                        gamma1=args.gamma1,
                                        betas=betas[:, k*s:(k+1)*s],
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
                errs.extend(errs_)
                cnt = sum(np.array(errs) <= args.E)
                if len(errs) > 20 and cnt < len(errs) - 20: break # early abort 
                gamma = np.maximum(gamma, np.array(gamma_))
            gamma = gamma[0]
            errs = np.array(errs)
            cnt = sum(np.array(errs) <= args.E)
            rhoE = (1 - proportion_confint(cnt, args.nrBetas, alpha=2 * args.alpha_E, method="beta")[0])
            max_err = errs.max()
        else:
            _, rhoE, max_err, gamma = next(args.Efile)
        t1 = time.time()
        print("max err:", max_err, rhoE)
        print("gamma", gamma)
        c, R = ksmooth(args,
                       model,
                       img_a,
                       sample_transformation,
                       args.E,
                       rhoE,
                       pre=pre)
        t2 = time.time()
        print(idx, label, pred_clean, pred_a, args.E, rhoE, gamma, max_err, c, R, t1 - t0, t2 - t1, file=logger, flush=True)
