import PIL
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import scipy.fftpack as fft
import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import compose
import scipy as sp
import scipy.ndimage
import scipy.signal
import imgaug as ia


def getGaussianFilter(filter_size, sigma):
    assert(filter_size % 2 == 1)
    G = np.zeros((filter_size, filter_size))
    c = filter_size // 2
    for i in range(c + 1):
        for j in range(c + 1):
            r = i * i + j * j
            G[c + i, c + j] = r
            G[c + i, c - j] = r
            G[c - i, c + j] = r
            G[c - i, c - j] = r
    G = np.exp(-G / sigma) / (np.pi * sigma)
    G = G / G.sum()
    return G


class Filter(nn.Module):
    def __init__(self, size, sigma, channels):
        super().__init__()
        self.channels = channels
        self.size = size
        self.c = size // 2
        self.sigma = sigma
        G = getGaussianFilter(size, sigma)
        G = np.repeat(G[np.newaxis, np.newaxis, :], self.channels, 0)
        G = torch.tensor(G, dtype=torch.float)
        self.G = nn.Parameter(G, requires_grad=False)

    #  so I can also use it as a transformer
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        shape = x.size()
        squeeze = False
        if len(shape) == 4:
            b, c, h, w = shape
        elif len(shape) == 3:
            c, h, w = shape
            b = 1
            x = x.unsqueeze(0)
            squeeze = True
        else:
            assert(False)
        
        assert(c == self.channels)
        out = F.conv2d(x, self.G, padding=self.c, groups=self.channels)
        bo, co, ho, wo = out.size()
        assert(bo == b)
        assert(co == c)
        assert(ho == h)
        assert(wo == w)
        if squeeze:
            out = out.squeeze(0)
        return out


def filterNP(x, size, sigma):
    G = getGaussianFilter(size, sigma)
    s = x.shape
    if len(s) == 2:
        return sp.ndimage.convole(x, G, mode='constant', cval=0)
    elif len(s) == 3:
        out = np.zeros(s)
        for i in range(s[2]):
            out[..., i] = sp.ndimage.convolve(x[..., i], G, mode='constant', cval=0)
        return out


def rotate(x, angle, resample=PIL.Image.BILINEAR, continous=False):
    if continous:
        order = {PIL.Image.NEAREST: 0,
                 PIL.Image.BILINEAR: 1}[resample]
        return ia.augmenters.Affine(rotate=angle, order=order)(image=x)
    else:
        return transforms.functional.affine(x, angle=angle, translate=(0, 0),
                                            scale=1, shear=(0, 0), resample=resample)


def translate(x, delta, resample=PIL.Image.BICUBIC, continous=False):
    if continous:
        assert False
    else:
        if isinstance(delta, np.ndarray):
            delta = delta.tolist()
        return transforms.functional.affine(x, angle=0, translate=delta,
                                            scale=1, shear=(0, 0), resample=resample)


def resize_crop(x, continous=False):
    if continous:
        x = ia.imresize_single_image(x, (224, 224), 'linear')
        x = ia.augmenters.CenterCropToFixedSize(height=224, width=224)(image=x)
        return x
    else:
        return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(x)


to_tensor = transforms.Compose([transforms.ToTensor()])


def lpfilter(img, bw):
    assert(bw % 2 == 0)
    ff = np.zeros(img.shape[:2])
    delta = bw // 2
    ff[0:delta, :] = 1
    ff[-delta:, :] = 1
    ff[:, 0:delta] = 1
    ff[:, -delta:] = 1
    out = np.zeros(img.shape)
    for c in range(3):
        fchannel = fft.fft2(img[:, :, c])
        fchannel = ff * fchannel
        out[:, :, c] = np.real(fft.ifft2(fchannel))
    return out

def T(t):
    def T_(in_):
        if isinstance(in_, np.ndarray):
            out_ = in_
            out_[out_ < t] = 0
            out_[out_ >= t] = 1
            return out_
        elif isinstance(in_, tuple) and len(in_) == 4:
            (x_corr, x_interpolate, diff, err_post_crop_l2) = in_
            x_corr_r_np_post = x_corr
            x_corr_r_np_post[x_corr_r_np_post < t] = 0
            x_corr_r_np_post[x_corr_r_np_post >= t] = 1
            
            x_interpolate_r_np_post = x_interpolate
            x_interpolate_r_np_post[x_interpolate_r_np_post < t] = 0
            x_interpolate_r_np_post[x_interpolate_r_np_post >= t] = 1            
            diff_post = x_corr_r_np_post - x_interpolate_r_np_post
            err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=np.inf)
            return x_corr_r_np_post, x_interpolate_r_np_post, diff_post, err_post_crop_l2_post
        else:
            assert False
    return T_

def Id():
    def Id_(in_):
        return in_
    return Id_


def LP(bw):
    def LP_(in_):
        if isinstance(in_, np.ndarray):
            return lpfilter(in_, bw)
        elif isinstance(in_, tuple) and len(in_) == 4:
            (x_corr, x_interpolate, diff, err_post_crop_l2) = in_
            x_corr_r_np_post = lpfilter(x_corr, bw)
            x_interpolate_r_np_post = lpfilter(x_interpolate, bw)
            diff_post = x_corr_r_np_post - x_interpolate_r_np_post
            err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=np.inf)
            return x_corr_r_np_post, x_interpolate_r_np_post, diff_post, err_post_crop_l2_post
        else:
            assert False
    return LP_

# def Blur(bw):
#     def Blur_(in_):
#         if isinstance(in_, np.ndarray):
#             return lpfilter(in_, bw)
#         elif isinstance(in_, tuple) and len(in_) == 4:
#             (x_corr, x_interpolate, diff, err_post_crop_l2) = in_
#             x_corr_r_np_post = lpfilter(x_corr, bw)
#             x_interpolate_r_np_post = lpfilter(x_interpolate, bw)
#             diff_post = x_corr_r_np_post - x_interpolate_r_np_post
#             err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=np.inf)
#             return x_corr_r_np_post, x_interpolate_r_np_post, diff_post, err_post_crop_l2_post
#         else:
#             assert False
#     return Blur_


def Blur(n, s):
    def B_(in_):
        if isinstance(in_, np.ndarray):
            return filterNP(in_, n, s)
        elif isinstance(in_, tuple) and len(in_) == 4:
            (x_corr, x_interpolate, diff, err_post_crop_l2) = in_
            x_corr_r_np_post = filterNP(x_corr, n, s)
            x_interpolate_r_np_post = filterNP(x_interpolate, n, s)
            diff_post = x_corr_r_np_post - x_interpolate_r_np_post
            err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=np.inf)
            return x_corr_r_np_post, x_interpolate_r_np_post, diff_post, err_post_crop_l2_post
        else:
            assert False
    return B_

def Round(a):
    def Round_(in_):
        if isinstance(in_, np.ndarray):
            return (np.floor_divide(x_corr*255, a) * a) / 255.0
        elif isinstance(in_, tuple) and len(in_) == 4:
            (x_corr, x_interpolate, diff, err_post_crop_l2) = in_
            x_corr_r_np_post = (np.floor_divide(x_corr*255, a) * a) / 255.0
            x_interpolate_r_np_post = (np.floor_divide(x_interpolate*255, a) * a) / 255.0
            diff_post = x_corr_r_np_post - x_interpolate_r_np_post
            err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=2)
            return x_corr_r_np_post, x_interpolate_r_np_post, diff_post, err_post_crop_l2_post
        else:
            assert False
    return Round_

# def JPEG(q):
#     def JPEG_(x_corr, x_interpolate,
#               diff, err_post_crop_l2):
#         buffer_x_corr = BytesIO()
#         x_corr.save(buffer_x_corr, "JPEG", quality=q)
#         buffer_x_corr.seek(0)
#         x_corr_r_post = Image.open(buffer_x_corr)

#         buffer_x_interpolate = BytesIO()
#         x_interpolate.save(buffer_x_interpolate, "JPEG", quality=q)
#         buffer_x_interpolate.seek(0)
#         x_interpolate_r_post = Image.open(buffer_x_interpolate)
                
#         x_corr_r_np_post = np.array(x_corr_r_post) / 255.0
#         x_interpolate_r_np_post = np.array(x_interpolate_r_post) / 255.0
#         diff_post = x_corr_r_np_post - x_interpolate_r_np_post
#         err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=2)
#         return x_corr_r_np_post, x_interpolate_r_np_post, diff_post, err_post_crop_l2_post
#     return JPEG_ 

def Wavelet(t):
    def Wavelet_(in_):
        (x_corr, x_interpolate, diff, err_post_crop_l2) = in_
        def f(x, th):
            out = np.zeros(x.shape)
            for c in range(3):
                cA, (cH, cV, cD) = pywt.dwt2(x[:, :, c], 'haar')
                cA[cA < th] = 0
                cH[cH < th] = 0
                cV[cV < th] = 0
                cD[cD < th] = 0
                out[:, :, c] =  pywt.idwt2((cA, (cH, cV, cD)), 'haar')
            return out
        x_corr_r_np_post = f(x_corr, t)
        x_interpolate_r_np_post = f(x_interpolate, t)
        diff_post = x_corr_r_np_post - x_interpolate_r_np_post
        err_post_crop_l2_post = np.linalg.norm(np.reshape(diff_post, (-1,)), ord=2)
        return x_corr_r_np_post, x_interpolate_r_np_post, diff_post, err_post_crop_l2_post    
    return Wavelet_


def get_vingette_mask(shape, offset=0, shape_type='circ'):
    assert len(shape) == 3
    if shape_type == 'circ':
        c, h, w = shape
        assert h == w
        mask = Image.new('I', (h, w))
        draw = ImageDraw.Draw(mask)
        lu = (0+offset, 0+offset)
        rd = (h-offset, h-offset)
        draw.ellipse([lu, rd], fill=(1))
        del draw
        mask = np.array(mask).astype(np.float32)
        mask = np.tile(np.expand_dims(mask, axis=0), (c, 1, 1))
        return mask
    elif shape_type == 'rect':
        offset = int(offset)
        mask = np.ones(shape)
        mask[:, :offset, :]  = 0
        mask[:, -offset:, :]  = 0
        mask[:, :, -offset:]  = 0
        mask[:, :, :offset]  = 0
        return mask

class Vingette:

    def __init__(self, shape, shape_type='circ', pt=False, cuda=False, batch_dim=True, transpose=False, offset=0):
        print(shape)
        print(offset)
        self.mask = get_vingette_mask(shape, offset=offset, shape_type='circ')
        self.pt = pt
        if pt:
            if batch_dim:
                self.mask = np.expand_dims(self.mask, 0)
            self.mask = torch.from_numpy(self.mask)
            self.masks = {}
            for i in range(torch.cuda.device_count()):
                d = torch.device(f"cuda:{i}")
                self.masks[d] = self.mask.clone().to(d)
            d = torch.device("cpu")
            self.masks[d] = self.mask.clone().to(d)
            if cuda:
                self.mask = self.mask.cuda()
        else:
            if transpose:
                self.mask = np.transpose(self.mask, (1, 2, 0))

    def __call__(self, x):
        if self.pt:
            return self.masks[x.device] * x
        else:
            return x * self.mask
        
def get_postprocessing(arg):
    if arg is not None and len(arg) > 0:
        e = eval(arg)
        if isinstance(e, list):
            e = compose(*e)
        return e
    else:
        return Id()
