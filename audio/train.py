from cox.utils import Parameters
import cox.store
import argparse
from robustness import model_utils, datasets, train, defaults
from robustness.attacker import AttackerModel
import torch
import os
import os.path
import librosa
import numpy as np
from transforms import scale_volume, shift
import sys
sys.path.append('./GCommandsPytorch')
from gcommand_loader import find_classes, make_dataset  # noqa: E402
sys.path.append('..')
from util import str2bool
from resnet import resnet50
from training_attack import GaussianL2Step, GaussianNoise  # noqa: E402


#load as in GCommandsPytorch
class CustomGCommandLoader(torch.utils.data.Dataset):

    def __init__(self, split, args):
        self.args = args
        self.path = os.path.join("./GCommandsPytorch/data/", split)
        path = "./GCommandsPytorch/data/train"
        classes, class_to_idx = find_classes(path)
        self.data = make_dataset(path, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.data[index]
        y, sr = librosa.load(path, sr=None)
        # transformation
        t = {'volume': scale_volume,
             'pitch_shift': shift,
             'none': lambda y_, sr_, a: (y, sr_)}[self.args.trafo]
        a_low = -self.args.param
        a_high = self.args.param
        a = np.random.uniform(a_low, a_high)
        y, sr = t(y, sr, a)
        y = y + self.args.sigmaN * np.random.randn(*y.shape)
        n_fft = int(sr * self.args.window_size) #512
        win_length = n_fft
        hop_length = int(sr * self.args.window_stride)

        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.args.window_type)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)

        # make all spects with the same dims
        if spect.shape[1] < self.args.max_len:
            pad = np.zeros((spect.shape[0], self.args.max_len - spect.shape[1]))
            spect = np.hstack((spect, pad))
        elif spect.shape[1] > self.args.max_len:
            spect = spect[:, :self.args.max_len]
        spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
        spect = torch.FloatTensor(spect)

        # z-score normalization
        if self.args.normalize:
            mean = spect.mean()
            std = spect.std()
            if std != 0:
                spect.add_(-mean)
                spect.div_(std)
        return spect, target

    def __len__(self):
        return len(self.data)


parser = argparse.ArgumentParser()
parser.add_argument('--trafo', choices=['volume', 'pitch_shift', 'none'], default='none', help='')
parser.add_argument('--param', type=float, default=1.0, help='')

parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--nr_workers', type=int, default=16, help='data loader worker')

parser.add_argument('--sigmaN', type=float, default=0.2, help='std for gaussian noise')
parser.add_argument('--lr', type=float, default=0.01, help='leanring rate')
parser.add_argument('--eps', type=float, default=1.0, help='epsilon for L2 PGD')
parser.add_argument('--epochs', type=int, default=90, help='nr of epochs')
parser.add_argument('--out-dir', type=str, required=True, help='output path')
parser.add_argument('--fade-in', type=int, default=1, help='fade in for eps')

# from GCommandsPytorch; don't touch
parser.add_argument('--window_size', default=.02, help='window size for the stft')
parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
parser.add_argument('--window_type', default='hamming', help='window type for the stft')
parser.add_argument('--max_len', default=101, type=int, help='maximal length of a window')
parser.add_argument('--normalize', default=False, help='boolean, wheather or not to normalize the spect')

args = parser.parse_args()
GaussianL2Step.sigma = 0
GaussianL2Step.clamp = False
GaussianNoise.sigma = 0

# loading data
train_dataset = CustomGCommandLoader('train', args)
valid_dataset = CustomGCommandLoader('valid', args)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.nr_workers,
                                           pin_memory=True)
val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.nr_workers,
                                        pin_memory=True)

class Wrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        
        def forward(self, x, **kwargs):
            return self.net(x)

class ID(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x, **kwargs):
            return x

model = resnet50(num_classes=30)
model = Wrapper(model)
d = argparse.Namespace()
d.mean = torch.tensor(0)
d.std = torch.tensor(1)
model = AttackerModel(model, d)
model.normalizer = ID()
model.attacker.normalize = ID()

# Create a cox store for logging
out_store = cox.store.Store(args.out_dir)

# Hard-coded base parameters
train_kwargs = {
    'out_dir': args.out_dir,
    'adv_train': 0,
    'constraint': GaussianNoise,
    'eps': args.eps,
    'attack_lr': args.eps,
    'lr': args.lr,
    'attack_steps': 1,
    'step_lr': 2000,
    'random_start': 1,
    'use_best': False,
    'epochs': args.epochs,
    'save_ckpt_iters': -1,  # best and last
    'eps_fadein_epochs': args.fade_in
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
# use Imagnet defaults
ds_class = datasets.DATASETS['imagenet']
train_args = defaults.check_and_fill_args(train_args,
                                          defaults.TRAINING_ARGS, ds_class)
train_args = defaults.check_and_fill_args(train_args,
                                          defaults.PGD_ARGS, ds_class)

# Train a model
train.train_model(train_args, model, (train_loader, val_loader), store=out_store)
