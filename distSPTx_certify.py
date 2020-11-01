from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing, setup_args_getE
from classify_ksmooth import ksmooth, setup_args_ksmooth, run_model, pre, sample_transformation
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
import geometrictools as gt
from statsmodels.stats.proportion import proportion_confint
from transformations import rotate, translate, get_postprocessing, Vingette
from util import get_interpolation, str2bool, split
from multiprocessing import Pool
import pandas as pd

def _sample_err(args_):
    args, x, alpha, beta, V, post = args_
    errs = []
    post = get_postprocessing(post)
    K = alpha.shape[0]
    L = alpha.shape[1]
    for k in range(K):
        err = 0
        for l in range(L):
            if args.transformation == 'rot':
                x_corr = rotate(x, alpha[k, l, 0] + beta[k, 0], args.interpolation)
                x_interpolate = rotate(rotate(x, alpha[k, l, 0], args.interpolation), beta[k, 0], args.interpolation)
            elif args.transformation == 'translation':
                x_corr = translate(x, (alpha[k, l, 0] + beta[k, 0], alpha[k, l, 1] + beta[k, 1]), args.interpolation)
                x_interpolate = translate(translate(x, (alpha[k, l, 0], alpha[k, l, 1]), args.interpolation), (beta[k, 0], beta[k, 1]), args.interpolation)
            else: assert(False)
            x_corr = pre(args, x_corr)
            x_interpolate = pre(args, x_interpolate)
            if args.dataset in ['mnist', 'fashionmnist']:
                x_corr = np.expand_dims(x_corr, axis=0)
                x_interpolate = np.expand_dims(x_interpolate, axis=0)
            x_corr_np = np.array(x_corr)
            x_interpolate_np = np.array(x_interpolate)
            x_corr_np_d = x_corr_np / 255.0
            x_interpolate_np_d = x_interpolate_np / 255.0
            x_corr_np_post, x_interpolate_np_post = post(x_corr_np_d), post(x_interpolate_np_d)
            if V is not None:
                x_corr_np_post = V(x_corr_np_post)
                x_interpolate_np_post = V(x_interpolate_np_post)
            diff_post = x_corr_np_post - x_interpolate_np_post
            err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=2)
            err = max(err, err_post_crop_l2_post)
        errs.append(err)
    return errs

    
def sample_err(args, img):
    s = np.array(img).shape
    if len(s) == 2:
        w, h = s
        c = 1
    else:
        w, h, c = s
    wh_out = w
    if args.resize_post_transform > 0:
        wh_out = args.resize_post_transform
    if args.center_crop_post_transform > 0:
        wh_out = args.center_crop_post_transform 
    if args.radiusDecrease > 0:
        V = Vingette((c, wh_out, wh_out), 'circ', batch_dim=False, transpose=True, offset=args.radiusDecrease)
    else:
        V = None
    if args.filter_sigma > 0:
        post = f"Blur({args.filter_size}, {args.filter_sigma})"
    else:
        post = "Id()"

    #setup sampling
    alpha = np.random.uniform(args.gamma0, args.gamma1, size=(args.nrBetas, args.nrSamples, 1 if args.transformation == 'rot' else 2))
    beta = np.random.normal(0, args.sigma_gamma, size=(args.nrBetas, 1 if args.transformation == 'rot' else 2))
    Is = split(list(range(args.nrBetas)), args.threads)
    _args = []
    for I in Is:
        alphaI = alpha[I, ...]
        betaI = beta[I, ...]
        _args.append((args, img, alphaI, betaI, V, post))

    #sample
    with Pool(args.threads) as p:
        errs = p.map(_sample_err, _args)

    #flatten
    errs = [e for es in errs for e in es]
    return errs

def compute_err(args, img):
    print('Calculating E')
    if args.resize > 0:
        img = TF.resize(img, args.resize)
        img = TF.center_crop(img, args.resize)

    img = np.array(img)   
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    assert(img.dtype == np.uint8)
    img = img.astype(dtype=np.float32)/255.0

    if args.transformation == 'rot':
        betas = np.random.normal(0, args.sigma_gamma, args.nrBetas)
    elif args.transformation == 'trans':
        betas = np.random.normal(0, args.sigma_gamma, (2, args.nrBetas))
    errs = []
    s = args.betaSplit if args.betaSplit > 0 else args.nrBetas
    k = 0
    while len(errs) < args.nrBetas:
        errs_, _ = gt.getE(image=img,
                                transformation=args.transformation,
                                targetE=args.target_err,
                                stopE=args.stop_err,
                                gamma0=args.gamma0,
                                gamma1=args.gamma1,
                                betas=betas,
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
    return errs

def get_err(args, img):
    if args.Efile is not None:
        return next(args.Efile)
    if args.sampleE:
        errs = sample_err(args, img)
    else:
        errs = compute_err(args, img)

    err = max(errs[:args.guess_E_samples]) * args.guess_E_mult + args.guess_E_add
    cnt = sum(np.array(errs[args.guess_E_samples:]) <= err)
    rho = proportion_confint(cnt, len(errs[args.guess_E_samples:]),
                             alpha=args.alpha_E,
                             method="beta")[0]
    return err, (1-rho)


parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_ksmooth(parser)
parser = setup_args_getE(parser)
parser.add_argument('--Efile', type=str, default=None, help='load E and rho from file')
parser.add_argument('--betaSplit', type=int, default=-1, help='split beta in chunks for error bounding; higher numbers make the error computation faster; lower numbers more precise; -1 diables it and considers all errors together')
parser.add_argument('--alpha-E', type=float, default=0.001, help='confidence for E\rhoE estimate')
parser.add_argument('--sampleE', type=str2bool, default=False, help='use sampling to determine E')
parser.add_argument('--nrSamples', type=int, default=100, help='number of attacker angles to sample for each gamma')
parser.add_argument('--guess-E-samples', type=int, default=100, help='number of beta samples to use, to guess E')
parser.add_argument('--guess-E-mult', type=float, default=1.1, help='guess E by multiplying the maximally observed error on guess-E-samples with this value')
parser.add_argument('--guess-E-add', type=float, default=0, help='guess E by adding this to the maximally observed error on guess-E-samples, after applying guess-E-mult')


args = parser.parse_args()
setup(args)

if args.Efile is not None:
    df = pd.read_csv(args.Efile, skiprows=2, sep=' ', names=["index", "label", "predicted", "label_smooth", "radius", "E", "rho", "time", "tE"])
    e_rho = np.array(df[['E', 'rho']])
    n = e_rho.shape[0]
    def gen():
        for i in range(n):
            e, rho = e_rho[i, 0], e_rho[i, 1]
            yield (e, rho)
    args.Efile = gen()
    for i in range(args.Nstart): next(args.Efile)

args.gpu = 0 if args.use_cuda else -1
model = get_basemodel(args)
data = get_data(args)
logger = get_logger(args, __file__)

print(args, file=logger)
args.interpolation = {'nearest': PIL.Image.NEAREST,
                      'bilinear': PIL.Image.BILINEAR,
                      'bicubic': PIL.Image.BICUBIC}[args.interpolation]

print("index", "label", "predicted", "label_smooth", "radius", "E", "rho", "time", "tE", file=logger)
for idx, d in enumerate(data):
    print()
    img, label = d

    if args.resize > 0:
        img = TF.resize(img, args.resize)

    t0 = time.time()
    E, rho = get_err(args, img)
    print('E', E, 'rho', rho)
    tE = time.time() - t0
        
    pred_clean = run_model(args, model, [img], pre=pre).argmax(dim=1).item() if model is not None else None
    t0 = time.time()
    c, R = ksmooth(args,
                    model,
                    img,
                    sample_transformation,
                    E,
                    rho,
                    pre=pre)
    t1 = time.time()
    print(idx+args.Nstart, label, pred_clean, c, R, E, rho, t1 - t0, tE, file=logger, flush=True)

