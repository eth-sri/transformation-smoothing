import PIL
import argparse
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import os
import numpy as np
from transformations import rotate, translate, get_postprocessing, Vingette
from util import get_interpolation, str2bool, split
from data import get_dataset, ImagenetSizeLoader
from torchvision import transforms
import torchvision.transforms.functional as F

def run(k, args, xs, conn):
    np.random.seed(1000 * args.seed + k)
    I = get_interpolation(args.interpolation)
    train = get_dataset(args, 'train')
    resize_crop = transforms.Compose([transforms.Resize(args.resize_crop_resize) if args.resize_crop_resize > 0 else lambda y: y,
                                      transforms.CenterCrop(args.resize_crop_crop)  if args.resize_crop_crop > 0 else lambda y: y])
    if args.vingette != 'none':
        vingette_offset = {'cifar': 2, 'GTSRB': 2}.get(args.dataset, 0)
        V = Vingette((3, args.resize_crop_crop, args.resize_crop_crop), args.vingette, batch_dim=False, transpose=True, offset=vingette_offset)
        m = V.mask[:, :, 0]
    else:
        args.vingette = None
    post = get_postprocessing(args.post)
    for j, i in enumerate(xs):
        x = train[i][0]
        e = lambda y: (1+0.5**2)**y
        
        if args.scale_size > 0:
            x = transforms.Resize(args.scale_size)(x)
        for _ in range(args.L):
            if args.trafo == 'rotation':
                alpha1 = np.random.uniform(-args.alpha_min_max, args.alpha_min_max)
            elif args.trafo == 'translation':
                alpha1 = np.random.uniform(-args.alpha_min_max, args.alpha_min_max)
                alpha2 = np.random.uniform(-args.alpha_min_max, args.alpha_min_max)
            else:
                False
            for _ in range(args.K):
                pass
                if args.trafo == 'rotation':
                    beta1 = np.random.normal(0, args.beta_std)
                    x_corr = rotate(x, alpha1 + beta1, I)
                    x_interpolate = rotate(rotate(x, alpha1, I), beta1, I)
                elif args.trafo == 'translation':
                    beta1 = np.random.normal(0, args.beta_std)
                    beta2 = np.random.normal(0, args.beta_std)
                    x_corr = translate(x, (alpha1 + beta1, alpha2 + beta2), I)
                    x_interpolate = translate(translate(x, (alpha1, alpha2), I), (beta1, beta2), I)
                else:
                    False

                if args.dataset not in ['mnist', 'fashionmnist']:
                    x_corr_r = resize_crop(x_corr)
                    x_interpolate_r = resize_crop(x_interpolate)
                else:
                    x_corr_r = np.expand_dims(x_corr, axis=0)
                    x_interpolate_r = np.expand_dims(x_interpolate, axis=0)

                x_corr_r_np = np.array(x_corr_r)
                x_interpolate_r_np = np.array(x_interpolate_r)
                x_corr_r_np_d = x_corr_r_np / 255.0
                x_interpolate_r_np_d = x_interpolate_r_np / 255.0

                x_corr_r_np_post, x_interpolate_r_np_post = post(x_corr_r_np_d), post(x_interpolate_r_np_d)
                if args.vingette:
                    x_corr_r_np_d = V(x_corr_r_np_d)
                    x_interpolate_r_np_d = V(x_interpolate_r_np_d)
                    x_corr_r_np_post = V(x_corr_r_np_post)
                    x_interpolate_r_np_post = V(x_interpolate_r_np_post)
                diff = x_corr_r_np_d - x_interpolate_r_np_d
                diff_post = x_corr_r_np_post - x_interpolate_r_np_post

                err_post_crop_l2 = np.linalg.norm(np.reshape(diff, (-1,)), ord=2)
                err_post_crop_linf = np.linalg.norm(np.reshape(diff, (-1,)), ord=np.inf)

                err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=2)
                err_post_crop_linf_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=np.inf)

                if args.show and err_post_crop_l2_post > 0.5:
                    print(alpha1, beta1, err_post_crop_l2, err_post_crop_l2_post, err_post_crop_linf)

                    fig, axs = plt.subplots(2, 3, squeeze=False)
                    axs[0][0].imshow(x_corr_r_np, vmin=0, vmax=1)
                    axs[1][0].imshow(x_interpolate_r_np, vmin=0, vmax=1)
                    axs[0][1].imshow(x_corr_r_np_post, vmin=0, vmax=1)
                    axs[1][1].imshow(x_interpolate_r_np_post, vmin=0, vmax=1)
                    axs[0][2].imshow(diff_post)
                    axs[1][2].imshow(np.abs(diff_post))
                    plt.show()

                out = [i, alpha1, beta1, err_post_crop_l2, err_post_crop_l2_post, err_post_crop_linf, err_post_crop_linf_post]
                if args.dataset == 'mnist':
                    diff = np.transpose(diff, (1, 2, 0))
                    diff_post = np.transpose(diff_post, (1, 2, 0))
                h, w, c = diff.shape
                masks = []

                if args.split == 'lr':
                    mL = np.zeros((h, w, c))
                    mL[:(h//2), :, :] = 1
                    mR = np.zeros((h, w, c))
                    mR[(h//2):, :, :] = 1
                    masks = [mL, mR]
                elif args.split == '4':
                    mUL = np.zeros((h, w, c))
                    mUL[:(h//2), :(w//2), :] = 1
                    mUR = np.zeros((h, w, c))
                    mUR[(h//2):, :(w//2), :] = 1
                    mDL = np.zeros((h, w, c))
                    mDL[:(h//2), (w//2):, :] = 1
                    mDR = np.zeros((h, w, c))
                    mDR[(h//2):, (w//2):, :] = 1
                    masks = [mUL, mUR, mDL, mDR]
                elif args.split == '3':
                    n = 12
                    m0 = np.zeros((h, w, c))
                    m1 = np.zeros((h, w, c))
                    m2 = np.zeros((h, w, c))
                    m0[:n, :, :] = 1
                    m1[-n:, :, :] = 1
                    m2[n:-n, :, :] = 1                
                    masks = [m0, m1, m2]
                elif args.split == 'color':
                    mR = np.zeros((h, w, c))
                    mG = np.zeros((h, w, c))
                    mB = np.zeros((h, w, c))
                    mR[:, :, 0] = 1
                    mG[:, :, 1] = 1
                    mB[:, :, 2] = 1                
                    masks = [mR, mG, mB]
                for m in masks:
                    diffM = m * diff
                    diff_postM = m * diff_post
                    err_post_crop_l2M = np.linalg.norm(np.reshape(diffM, (-1,)), ord=2)
                    err_post_crop_linfM = np.linalg.norm(np.reshape(diffM, (-1,)), ord=np.inf)
                    err_post_crop_l2_postM = np.linalg.norm(np.reshape(diff_postM, (-1,)), ord=2)
                    err_post_crop_linf_postM = np.linalg.norm(np.reshape(diff_postM, (-1,)), ord=np.inf)                    
                    out.extend([err_post_crop_l2M, err_post_crop_linfM, err_post_crop_l2_postM, err_post_crop_linf_postM])
                conn.send(out)
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['imagenet', 'restricted_imagenet', 'mnist', 'fashionmnist', 'cifar', 'GTSRB'], default='imagenet', help='dataset')
    parser.add_argument('-N', type=int, default=0,
                        help='number of images from the training data')
    parser.add_argument('-K', type=int, default='100', help='number of smoothing angles samples per image')
    parser.add_argument('-L', type=int, default='1', help='number of attacker angle alpha to sample')
    parser.add_argument('--seed', type=int, default='0', help='seed')
    parser.add_argument('--trafo', choices=['rotation', 'translation'],
                        default='rotation', help='rotation or translation as transformation')
    parser.add_argument('--interpolation', choices=['nearest', 'bilinear', 'bicubic'],
                        default='bilinear', help='interpolation method')
    parser.add_argument('--alpha_min_max', type=float, default='8', help='attacker angle')
    parser.add_argument('--beta_std', type=float, default='15', help='standard deviation for beta')
    parser.add_argument('-p', type=int, default=16, help='processors')
    parser.add_argument('--min_size', type=int, default=0,
                        help='minimum size of the shorter size of the image')
    parser.add_argument('--scale_size', type=int, default=0,
                        help='resize image so that  the shorter side has this length before applying rotations')
    parser.add_argument('--show', type=str2bool, default=False, help='show images')
    parser.add_argument('--vingette', choices=['none', 'circ', 'rect'], default=False, help='vingette')
    parser.add_argument('--write', type=str2bool, default=True, help='write to file')
    parser.add_argument('--post', type=str, default='Id()', help='post processing')
    parser.add_argument('--resize-crop-resize', type=int, default=256, help='processors')
    parser.add_argument('--resize-crop-crop', type=int, default=224, help='processors')
    parser.add_argument('--split', choices=['none', 'lr', '4', '3', 'color'], default='none', help='processors')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print('loading dataset')
    if args.min_size > 0:
        if args.dataset == 'imagenet':
            isl = ImagenetSizeLoader(mp.cpu_count(), 'train')
            I = isl.get_indices(args.min_size)
        elif args.dataset == 'mnist':
            if args.min_size > 28:
                I = []
            else:
                ds = get_dataset(args, 'train')
                I = list(range(len(ds)))
        elif args.dataset == 'cifar':
            if args.min_size > 32:
                I = []
            else:
                ds = get_dataset(args, 'train')
                I = list(range(len(ds)))
        elif args.dataset == 'restricted_imagenet':
            raise NotImplementedError
    else:
        ds = get_dataset(args, 'train')
        I = list(range(len(ds)))

    if args.N == 0:
        N = len(I)
    else:
        N = min(args.N, len(I))
    samples = random.sample(list(I), N)
    sample_chunks = list(split(samples, args.p))

    pipes = [mp.Pipe() for i in range(args.p)]
    ps = [mp.Process(target=run, args=(i, args, sample_chunks[i], pipes[i][1]))
          for i in range(args.p)]
    recievers = [p for p, _ in pipes]

    for p in ps:
        p.start()

    os.makedirs("sampling", exist_ok=True)
    post = args.post
    for c in ['(', ')', '[', ']', ',', ' ', '.']:
        post = post.replace(c, '')
    fn = f"{args.dataset}_{args.trafo}_{args.interpolation}_{N}_{args.K}_{args.alpha_min_max}_{args.beta_std}_{args.min_size}_{args.scale_size}_{post}_{args.seed}_{args.split}{'_v' if args.vingette else ''}.csv"
    fn = os.path.join('sampling', fn)
    print(fn)
    f = open(fn, "w") if args.write else None
    R = 0
    M = N * args.K
    with tqdm(total=M) as pbar:
        while R < M:
            results = mp.connection.wait(recievers, timeout=1)
            for r in results:
                res_tuple = r.recv()
                if f:
                    print(", ".join(map(str, res_tuple)), file=f)
                R += 1
                pbar.update(1)
        if f:
            f.flush()
    for r in recievers:
        r.close()
    for p in ps:
        p.join()
    if f:
        f.close()
