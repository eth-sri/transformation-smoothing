import time
import torch.multiprocessing
import argparse
import torch
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from datetime import datetime
import random
import re
import os
import numpy as np
from transforms import scale_volume
from robustness import model_utils, datasets, train
from robustness.attacker import AttackerModel
import dill
import librosa
import glob
sys.path.append('./GCommandsPytorch')
from gcommand_loader import find_classes, make_dataset  # noqa: E402
sys.path.append('..')
import resnet as myresnet
from util import str2bool, Logger
from classify_utils import get_logger, setup_args, setup, setup_args_preprocessing

from classify_ksmooth import ksmooth, setup_args_ksmooth, run_model

# load data as in GCommandsPytorch
def to_mfcc(y, sr, args):
    n_fft = int(sr * args.window_size) #512
    win_length = n_fft
    hop_length = int(sr * args.window_stride)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=args.window_type)
    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)

    if spect.shape[1] < args.max_len:
        pad = np.zeros((spect.shape[0], args.max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > args.max_len:
        spect = spect[:, :args.max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)
    return spect


def pre(args, x):
    data, sr = x
    x = to_mfcc(data, sr, args)
    return x


def sample_transformation(args, x):
    data, sr = x
    return scale_volume(data, sr, args.sigma_gamma * np.random.randn())


class Loader(torch.utils.data.Dataset):

    def __init__(self, split):
        self.path = os.path.join("./GCommandsPytorch/data/", split)
        path = "./GCommandsPytorch/data/train"
        classes, class_to_idx = find_classes(path)
        self.data = make_dataset(path, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        path, target = self.data[index]
        y, sr = librosa.load(path, sr=None)
        return (y, sr), target

class RobustnessModelOuterWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)[0]
        return y

class RobustnessModelInnerWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(x)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = setup_args(parser)
    parser = setup_args_ksmooth(parser)
    parser = setup_args_preprocessing(parser)

    # from GCommandsPytorch; don't touch
    parser.add_argument('--window_size', default=.02, help='window size for the stft')
    parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
    parser.add_argument('--window_type', default='hamming', help='window type for the stft')
    parser.add_argument('--max_len', default=101, type=int, help='maximal length of a window')
    parser.add_argument('--normalize', default=False, help='boolean, wheather or not to normalize the spect')

    args = parser.parse_args()
    setup(args)
    logger = get_logger(args, __file__)
    print(args, file=logger)

    path = glob.glob(os.path.join(args.model, '**', 'checkpoint.pt.best'))[0]
    model = myresnet.resnet50(num_classes=30)
    model = RobustnessModelInnerWrapper(model)
    d = argparse.Namespace()
    d.mean = torch.tensor(0)
    d.std = torch.tensor(1)
    model = AttackerModel(model, d)
    checkpoint = torch.load(path, pickle_module=dill)
    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'

    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]:v for k, v in sd.items()}
    sd['normalizer.new_mean'] = torch.tensor([[0]])
    sd['attacker.normalize.new_mean'] = torch.tensor([[0]])
    sd['normalizer.new_std'] = torch.tensor([[1]])
    sd['attacker.normalize.new_std'] = torch.tensor([[1]])
    model.load_state_dict(sd)
    model = RobustnessModelOuterWrapper(model)

    model.to(args.device)
    model = torch.nn.DataParallel(model)
    model = model.eval().to(args.device)

    ds = Loader('valid')
    K = len(ds)
    samples = list(range(K))
    random.shuffle(samples)
    data = [ds[i] for i in samples[:args.N]]

    print("index", "label", "predicted", "label_smooth", "radius", "time", file=logger)
    for idx, d in enumerate(data):
        print()
        x, label = d

        pred_clean = run_model(args, model, [x], pre=pre).argmax(dim=1).item()
        
        if pred_clean != label:
            print(idx, label, pred_clean, None, None, None, file=logger)
        else:
            t0 = time.time()
            c, R = ksmooth(args,
                           model,
                           x,
                           sample_transformation,
                           args.E,
                           args.rhoE,
                           pre=pre)
            t1 = time.time()
            print(idx, label, pred_clean, c, R, t1 - t0, file=logger)
