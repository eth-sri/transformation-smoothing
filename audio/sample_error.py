import sys
import numpy as np
import random
import argparse
from operator import itemgetter
import multiprocessing as mp
from tqdm import tqdm
import os
import librosa
from transforms import scale_volume, shift
sys.path.append('..')
from util import str2bool, split, lmap  # noqa: E402
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


def spect_tranform(y, sr, window_size=.02, window_stride=.01, window='hamming', max_len=101):
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    spect_log = np.log1p(spect)
    return spect, spect_log


def transform(y, sr, a, b, args):
    """
    Parameters
    ----------
    y : actually sampled data
    sr : sample
    a : attacker parameter
    b : smoothing parameter
    args : args struct

    Returns
    -------
    difference between interpolated and non-interpolated data
    """
    t = {'volume': scale_volume,
         'pitch_shift': shift}[args.trafo]
    y_a, _ = t(y, sr, a)
    y_a_b, _ = t(y_a, sr, b)
    y_ab, _ = t(y, sr, a + b)
    #assert np.all(y_a_b < 32768)
    #assert np.all(y_ab < 32768)
    #y_ab /= 32768
    #y_a_b /= 32768
    diff = y_a_b - y_ab
    s_y_a_b, l_s_y_a_b = spect_tranform(y_a_b, sr)
    s_y_ab, l_s_y_ab = spect_tranform(y_ab, sr)
    spect_diff = s_y_a_b - l_s_y_a_b
    log_spect_diff = s_y_ab - l_s_y_ab
    return diff, spect_diff, log_spect_diff


def run(k, args, xs, conn):
    """
    Parameters
    ----------
    k : index of thread
    args : struct of all arguments
    xs: list of ids (int) of data points
    conn : pipe to the main thread

    function is run as a thread
    iterates though all x in xs and samples errors
    and reports them back to the main thread 
    """
    np.random.seed(1000 * args.seed + k)
    data = lmap(itemgetter(0), get_dataset())
    a_min = -args.alpha_min_max
    a_max = args.alpha_min_max
    for i in xs:
        path = data[i]
        y, sr = librosa.load(path, sr=None)
        for _ in range(args.K):
            a = np.random.uniform(a_min, a_max)
            b = np.random.normal(0, args.beta_std)
            diff, spect_diff, l_spect_diff = transform(y, sr, a, b, args)
            l1 = np.linalg.norm(diff, ord=1)
            l2 = np.linalg.norm(diff, ord=2)
            linf = np.linalg.norm(diff, ord=np.inf)
            ret = [i, a, b, l1, l2, linf]
            s_l1 = np.linalg.norm(np.ravel(spect_diff), ord=1)
            s_l2 = np.linalg.norm(np.ravel(spect_diff), ord=2)
            s_linf = np.linalg.norm(np.ravel(spect_diff), ord=np.inf)
            l_s_l1 = np.linalg.norm(np.ravel(l_spect_diff), ord=1)
            l_s_l2 = np.linalg.norm(np.ravel(l_spect_diff), ord=2)
            l_s_linf = np.linalg.norm(np.ravel(l_spect_diff), ord=np.inf)
            ret.extend([s_l1, s_l2, s_linf, l_s_l1, l_s_l2, l_s_linf])
            conn.send(ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=51088,
                        help='number of images from the training data')
    parser.add_argument('-K', type=int, default='100', help='number of angle samples per image')
    parser.add_argument('--seed', type=int, default='0', help='seed')
    parser.add_argument('--trafo', choices=['pitch_shift', 'volume'],
                        default='volume', help='transformation')
    parser.add_argument('--alpha_min_max', type=float, default='1.2',
                        help='attacker parameter between 1/x and x')
    parser.add_argument('--beta_std', type=float, default='0.3', help='standard deviation for beta')
    parser.add_argument('-p', type=int, default=16, help='processors')
    parser.add_argument('--write', type=str2bool, default=True, help='write to file')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    data = get_dataset()
    L = len(data)
    N = min(L, args.N)
    I = list(range(L))

    samples = list(random.sample(I, N))
    sample_chunks = list(split(samples, args.p))

    pipes = [mp.Pipe() for i in range(args.p)]
    ps = [mp.Process(target=run, args=(i, args, sample_chunks[i], pipes[i][1]))
          for i in range(args.p)]
    recievers = [p for p, _ in pipes]

    for p in ps:
        p.start()

    os.makedirs("sampling", exist_ok=True)
    fn = f"{args.trafo}_{N}_{args.K}_{args.alpha_min_max}_{args.beta_std}_{args.seed}.csv"
    fn = os.path.join('sampling', fn)
    f = open(fn, "w") if args.write else sys.stdout
    R = 0
    M = N * args.K
    with tqdm(total=M) as pbar:
        while R < M:
            results = mp.connection.wait(recievers, timeout=1)
            for r in results:
                res_tuple = r.recv()
                print(", ".join(map(str, res_tuple)), file=f)
                R += 1
                pbar.update(1)
        f.flush()
    for r in recievers:
        r.close()
    for p in ps:
        p.join()
    if args.write:
        f.close()
