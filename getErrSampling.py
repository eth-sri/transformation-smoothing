from classify_utils import setup_args, setup, get_basemodel, get_data, get_logger, setup_args_preprocessing, setup_args_getE
from classify_ksmooth import ksmooth, setup_args_ksmooth, run_model, pre
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
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

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
                x_corr_np = np.expand_dims(np.array(x_corr), axis=-1)
                x_interpolate_np = np.expand_dims(np.array(x_interpolate), axis=-1)
            else:
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
    errs = _sample_err((args, img, alpha, beta, V, post))
    return errs


def run(k, args, Is, imgs, conn):
    np.random.seed(1000 * args.seed + k) 
    for idx, img in zip(Is, imgs):
        if args.resize > 0:
            img = TF.resize(img, args.resize)
        errs = sample_err(args, img)
        conn.send((idx, errs))
    conn.close()

parser = argparse.ArgumentParser()
parser = setup_args(parser)
parser = setup_args_preprocessing(parser)
parser = setup_args_ksmooth(parser)
parser = setup_args_getE(parser)
parser.add_argument('--nrSamples', type=int, default='100', help='number of attacker angles to sample')

args = parser.parse_args()
setup(args)

args.gpu = 0 if args.use_cuda else -1
data = get_data(args, split='train')
logger = get_logger(args, __file__)
args.interpolation = {'nearest': PIL.Image.NEAREST,
                      'bilinear': PIL.Image.BILINEAR,
                      'bicubic': PIL.Image.BICUBIC}[args.interpolation]


I = list(range(len(data)))
Is = list(split(I, args.threads))



pipes = [mp.Pipe() for i in range(args.threads)]
ps = [mp.Process(target=run, args=(i, args, Is[i], [data[j][0] for j in Is[i]], pipes[i][1]))
        for i in range(args.threads)]
recievers = [p for p, _ in pipes]

print("Starting")
for p in ps:
    p.start()

print("index", "err", file=logger, flush=True)
i = 0
with tqdm(total=args.N) as pbar:
    while i < args.N:
        results = mp.connection.wait(recievers, timeout=1)
        for r in results:
            idx, errs = r.recv()
            i += 1 
            for err in errs:
                print(idx+args.Nstart, err, file=logger, flush=True)
            logger.flush()
            pbar.update(1)
for r in recievers:
    r.close()
for p in ps:
    p.join()
