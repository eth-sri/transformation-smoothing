import PIL
import PIL.Image
from functional import compose
import numpy as np
import argparse


lmap = compose(list, map)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def split(a, n):
    """
    Splits a list into n parts of approx. equal length
    from https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def str2FloatOrNone(v):
    if v.lower() == 'none':
        return None
    try:
        return float(v)
    except ValueError:
        raise argparse.ArgumentTypeError('Float or none value expected.')


def torch_image_to_PIL(img):
    img = img.cpu().numpy()
    if len(img.shape) == 4:
        img = img[0, ...]
    elif len(img.shape) == 3:
        pass
    else:
        assert False
    img = 255 * np.transpose(img, (1, 2, 0))
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(img)


class Logger(object):
    def __init__(self, filename, stdout):
        self.terminal = stdout
        if filename is not None:
            self.log = open(filename, "a")
        else:
            self.log = None

    def write(self, message):
        self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if self.log is not None:
            self.log.flush()


def get_interpolation(i):
    return getattr(PIL.Image, i.upper())
