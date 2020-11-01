import sys
import librosa
import numpy as np
import random
import argparse
from operator import itemgetter
import multiprocessing as mp
from tqdm import tqdm
import os
sys.path.append('..')
from util import str2bool, split, lmap  # noqa: E402
import io
import soundfile as sf
import tqdm
sys.path.append('./GCommandsPytorch')
from gcommand_loader import find_classes, make_dataset  # noqa: E402

def get_dataset():
    """
    Returns
    -------
    a dataset which is a list of tuples (path, label)
    """
    path = "./GCommandsPytorch/data/train"
    classes, class_to_idx = find_classes(path)
    data = make_dataset(path, class_to_idx)
    return data

def round(y, sr):
    bo = io.BytesIO()
    bo.name = "foo.wav"
    sf.write(bo, y, sr, 'PCM_16')
    bo.seek(0)
    y, sr = librosa.load(bo, sr=None)
    return y, sr

def run(arg_tuple):
    args, y, sr, a_, betas_ = arg_tuple
    delta = 1.0/(2**15)
    a_min = -args.alpha_min_max
    a_max = args.alpha_min_max
    
    out = []
    for j, a in enumerate(a_):
        gamma = 10**(a / 20)
        y_a_pre_round = gamma * y # attacked signal
        y_a, _ = round(y_a_pre_round, sr)

        y_a_l = y_a
        y_a_u = y_a + delta
        y_a_u[y_a >= 0.9999695] = max(1, 10**(a_max/ 20))
        y_a_l[y_a == -1] = min(-1, -1 * 10**(a_max/ 20))
        assert np.all(y_a_l <= y_a_u)
        
        if not np.all(y_a_l <= y_a_pre_round):
            print(y_a_l[y_a_l > y_a_pre_round])
            print(y_a_pre_round[y_a_l > y_a_pre_round])
        if not np.all(y_a_pre_round <= y_a_u):
            print(y_a_pre_round[y_a_pre_round > y_a_u])
            print(y_a[y_a_pre_round > y_a_u])            
            print(y_a_u[y_a_pre_round > y_a_u])

        assert(np.all(y_a_l <= y_a_pre_round))
        assert(np.all(y_a_pre_round <= y_a_u))
        
        I, step = np.linspace(a_min, a_max, num=20000, endpoint=False, retstep=True)
        err = 0
        betas = betas_[j] #np.random.normal(0, args.beta_std, 100)
        for start in I.tolist(): #tqdm.tqdm(I.tolist()):
            end = start + step
            assert(start < end)
            gamma_l = 10**(start / 20)
            gamma_u = 10**(end / 20)
            assert(0 < gamma_l < gamma_u)
            gamma_inv_l = 1.0 / gamma_u
            gamma_inv_u = 1.0 / gamma_l

            a = gamma_inv_l * y_a_l
            b = gamma_inv_u * y_a_l
            c = gamma_inv_l * y_a_u
            d = gamma_inv_u * y_a_u
            y_inv_l = np.minimum(np.minimum(a, b),
                                 np.minimum(c, d))
            y_inv_u = np.maximum(np.maximum(a, b),
                                 np.maximum(c, d))

            y_inv_l = np.ceil(y_inv_l * (2**15)) / (2**15)
            y_inv_u = np.floor(y_inv_u * (2**15)) / (2**15)

            empty = not np.all(y_inv_l <= y_inv_u)
            #assert(np.all(y_inv_l <= y_inv_u))

            if gamma_l <= gamma <= gamma_u:
                assert(not empty)
                idx = np.logical_or((y_inv_l > y), (y > y_inv_u))
                assert(idx.sum() == 0)

                
            if not empty:
                errs = []
                for beta in betas:
                    y_ab_l = 10**((start + beta) / 20) * y_inv_l
                    y_ab_u = 10**((end + beta) / 20) * y_inv_u

                    y_a_b_l = 10**((beta) / 20) * y_a_l
                    y_a_b_u = 10**((beta) / 20) * y_a_u

                    diff_l = y_ab_l - y_a_b_u
                    diff_u = y_ab_u - y_a_b_l
                    square_diff = np.maximum(diff_l * diff_l,
                                             diff_u * diff_u)
                    l2_u = np.sqrt(square_diff.sum())
                    errs.append(l2_u)
                err = max(err, max(errs))
        out.append(err)

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=50,
                        help='number of images from the training data')
    parser.add_argument('-K', type=int, default=1, help='number of angle samples per image')
    parser.add_argument('--seed', type=int, default='0', help='seed')
    parser.add_argument('--alpha_min_max', type=float, default='1.05')
    parser.add_argument('--beta_std', type=float, default='0.85', help='standard deviation for beta')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    
    data = get_dataset()
    L = len(data)
    N = min(L, args.N)
    I = list(range(L))
    samples = list(random.sample(I, N))
    
    data = lmap(itemgetter(0), get_dataset())

    a_min = -args.alpha_min_max
    a_max = args.alpha_min_max

    arg_tuples = []
    
    for i in samples:
        path = data[i]
        y, sr = librosa.load(path, sr=None)
        a_ = [np.random.uniform(a_min, a_max) for _ in range(args.K)]
        betas_ = [np.random.normal(0, args.beta_std, 100) for _ in range(args.K)]
        arg_tuples.append((args, y, sr, a_, betas_))
    with mp.Pool() as pool:
        errs = pool.map(run, arg_tuples)

    errs = [b for a in errs for b in a]
    with open('indiv_err_bound.txt', 'w') as log:
        for err in errs:
            print(err)
            print(err, file=log)
