from cox.utils import Parameters
import cox.store
from transformations import Vingette, Filter
from torchvision import transforms, models
from robustness.attacker import AttackerModel
import PIL
import argparse
from robustness import model_utils, datasets, train, defaults, loaders
from data import get_dataset
from util import str2bool
import os
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from training_attack import GaussianL2Step, GaussianNoise
from resnet import resnet18
from mnist_net import MNISTConvNet
from classify_utils import RobustnessModelInnerWrapper

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--nr-workers', type=int, default=16, help='data loader worker')
    parser.add_argument('--dataset', choices=['imagenet', "mnist", 'restricted_imagenet', 'GTSRB', 'cifar', 'fashionmnist' ], help='dataset')
    parser.add_argument('--sigmaN', type=float, default=0.5, help='std for gaussian noise')
    parser.add_argument('--eps', type=float, default=1.0, help='epsilon for L2 PGD')
    parser.add_argument('--attack', type=str2bool, default=True, help='use adversarial training')
    parser.add_argument('--vingette', choices=['none', 'circ', 'rect'], default='none', help='use vingetting')
    parser.add_argument('--vingette-offset', type=int, default=-1, help='radius for vingetting')
    parser.add_argument('--filter-sigma', type=float, default=0, help='gaussian blur filter sigma')
    parser.add_argument('--filter-size', type=int, default=0, help='gaussian blur filter size')
    parser.add_argument('--rotate', type=float, default=0, help='use rotation with +/- N degrees as data augmentaiton; values <= 0 disable it')
    parser.add_argument('--translate', type=float, default=0, help='use translation with +/- N percent iamgesize translation; values <= 0 disable it')
    parser.add_argument('--epochs', type=int, default=90, help='nr of epochs')
    parser.add_argument('--start-epoch', type=int, default=None, help='epoch to start from -- None defaults to 0 or to the value from resume')
    parser.add_argument('--out-dir', type=str, required=True, help='output path')
    parser.add_argument('--fade-in', type=int, default=1, help='fade in for eps')
    parser.add_argument('--resume-from', type=str, default=None, help='if given resume training from the given checkpoint')
    parser.add_argument('--crop-resize-resize', type=int, default=256, help='prior to classificaiton resize to this size')
    parser.add_argument('--crop-resize-crop', type=int, default=224, help='prior to classificaiton crop a subimage of this size')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--step-lr', type=int, default=30, help='spets after which to decrease lr by factor 10')
    parser.add_argument('--mnist-convnet', type=str2bool, default=False, help='use a convnet rather than resnet18')
    
    args = parser.parse_args()
    args.data_set = args.dataset
    GaussianL2Step.sigma = args.sigmaN
    GaussianNoise.sigma = args.sigmaN
    c = 1 if 'mnist' in args.dataset else 3
    if args.vingette != 'none':
        v = Vingette((c, args.crop_resize_crop, args.crop_resize_crop), args.vingette, pt=True, cuda=True, offset=args.vingette_offset)
        GaussianL2Step.vingette = v
        GaussianNoise.vingette = v

    args.out_dir = os.path.join('models', args.out_dir)
    args.data_dir = {'GTSRB': './ds/GTSRB',
                     'imagenet': './ds/imagenet',
                     'restricted_imagenet': './ds/imagenet',
                     'mnist': './ds/mnist',
                     'fashionmnist': './ds/fashionmnist',
                     'cifar': './ds/cifar'}[args.dataset]
    args.data_dir = os.path.abspath(args.data_dir)

    ts = []
    if args.rotate > 0:
        ts.append(transforms.RandomRotation(args.rotate, resample=PIL.Image.BILINEAR))
    if args.translate > 0:
        ts.append(transforms.RandomAffine(0, translate=(args.translate, args.translate), resample=PIL.Image.BILINEAR))
    if args.crop_resize_resize != 0:
        ts.append(transforms.Resize(args.crop_resize_resize, interpolation=PIL.Image.BILINEAR))
    if args.crop_resize_crop != 0:
        ts.append(transforms.RandomCrop(args.crop_resize_crop))
    ts.append(transforms.RandomHorizontalFlip())
    ts.append(transforms.ToTensor())
    if args.filter_sigma > 0:
        assert(args.filter_size > 0 and args.filter_size % 2 == 1)
        ts.append(Filter(args.filter_size, args.filter_sigma, c))
    if args.vingette != 'none':
        ts.append(Vingette((c, args.crop_resize_crop, args.crop_resize_crop), args.vingette, pt=True, batch_dim=False, offset=args.vingette_offset))       
    training_transform = transforms.Compose(ts)

    ts = []
    if args.crop_resize_resize != 0:
        ts.append(transforms.Resize(args.crop_resize_resize, interpolation=PIL.Image.BILINEAR))
    if args.crop_resize_crop != 0:
        ts.append(transforms.CenterCrop(args.crop_resize_crop))
    ts.append(transforms.ToTensor())
    if args.filter_sigma > 0:
        assert(args.filter_size > 0 and args.filter_size % 2 == 1)
        ts.append(Filter(args.filter_size, args.filter_sigma, c))
    if args.vingette != 'none':
        ts.append(Vingette((c, args.crop_resize_crop, args.crop_resize_crop), args.vingette, pt=True, batch_dim=False, offset=args.vingette_offset))
    test_transform = transforms.Compose(ts)

    if args.dataset in ['imagenet', 'restricted_imagenet', 'cifar']:
        ds_class = datasets.DATASETS[args.dataset]
        ds = ds_class(args.data_dir)
        ds.transform_train = training_transform
        ds.transform_test = test_transform
        m, params = model_utils.make_and_restore_model(arch=('resnet18' if args.dataset == 'cifar' else 'resnet50'),
                                                       dataset=ds, resume_path=args.resume_from,
                                                       parallel=False)

        if args.start_epoch is None:
            if args.resume_from is not None:
                args.start_epoch = params['epoch']
            else:
                args.start_epoch = 0
        train_loader, _ = ds.make_loaders(batch_size=args.batch_size, workers=args.nr_workers)
        _, val_loader = ds.make_loaders(batch_size=args.batch_size//2, workers=args.nr_workers)
    elif 'mnist' in args.dataset:
        ds_class = datasets.DATASETS['cifar'] # pretend we are cifar
        train_dataset = get_dataset(args, 'train', transform=training_transform)
        valid_dataset = get_dataset(args, 'test', transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.nr_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.nr_workers,
                                                pin_memory=True)

        if args.mnist_convnet:
            model = MNISTConvNet()
        else:
            model = resnet18(num_classes=10, color_channels=1)
        model = RobustnessModelInnerWrapper(model)
        d = argparse.Namespace()
        if args.dataset == 'mnist':
            d.mean = torch.tensor([0.1307])
            d.std = torch.tensor([0.3081])
        else: #fashionmnist
            d.mean = torch.tensor([0])
            d.std = torch.tensor([1])
        m = AttackerModel(model, d)
    elif args.dataset == 'GTSRB':
        ds_class = datasets.DATASETS['cifar']# pretend we are cifar
        ds = ds_class(args.data_dir)
        ds.mean = torch.tensor([0.3337, 0.3064, 0.3171])
        ds.std = torch.tensor([0.2672, 0.2564, 0.2629])
        transforms = (training_transform, test_transform)
        train_loader, val_loader = loaders.make_loaders(workers=args.nr_workers,
                                                        batch_size=args.batch_size,
                                                        transforms=transforms,
                                                        data_path=args.data_dir,
                                                        data_aug=True,
                                                        dataset="GTSRB",
                                                        label_mapping=None,
                                                        custom_class=None,
                                                        val_batch_size=args.batch_size,
                                                        shuffle_train=True,
                                                        shuffle_val=False)
        model = resnet18(num_classes=43, color_channels=3)
        model = RobustnessModelInnerWrapper(model)
        m = AttackerModel(model, ds)
    else:
        assert False

    # Create a cox store for logging
    out_store = cox.store.Store(args.out_dir)

    if args.start_epoch is not None: 
        args.lr = args.lr / 10**(args.start_epoch // args.step_lr)

    print(f"Starting at epoch {args.start_epoch}, running until epoch {args.epochs}; starting with lr {args.lr}")

    # Hard-coded base parameters
    train_kwargs = {
        'out_dir': args.out_dir,
        'adv_train': 1,
        'constraint': GaussianL2Step if args.attack else GaussianNoise,
        'eps': args.eps,
        'attack_lr': args.eps,
        'attack_steps': 1,
        'random_start': 1,
        'use_best': False,
        'epochs': (args.epochs - args.start_epoch) if args.start_epoch is not None else args.epochs,
        'save_ckpt_iters': -1, # best and last
        'eps_fadein_epochs': args.fade_in,
        'step_lr': args.step_lr,
        'lr': args.lr,
        'weight_decay': 1e-5
    }
    train_args = Parameters(train_kwargs)

    # Fill whatever parameters are missing from the defaults
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.TRAINING_ARGS, ds_class)
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.PGD_ARGS, ds_class)


    # ensure data parallel works
    torch.cuda.set_device(0)
    
    # Train a model
    train.train_model(train_args, m, (train_loader, val_loader), store=out_store)
