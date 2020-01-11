import numpy as np
import torch
from torch.nn import functional as F


def to_one_hot(tensor, n):
    one_hot = torch.zeros(tensor.size() + (n,))
    one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), 1.)
    return one_hot


def softplus(x, min_thresh=0.0):
    assert min_thresh <= 0
    # return f.softplus(x - min_thresh) + min_thresh
    eps = 1e-2
    h = np.log(np.exp(-min_thresh) - 1 + eps)
    return F.softplus(x + h) + min_thresh


def get_imgs_pad_value(imgs):
    """
    Hack to visualize boundaries between images with save_image().
    If the median border value of all images is 0, use white, otherwise
    black (which is the default)
    :param imgs: 4d tensor
    :return: padding value
    """
    imgs = imgs.mean(1)  # reduce to 1 channel
    d1 = imgs.size(1)
    d2 = imgs.size(2)
    borders = list()
    borders.append(imgs[:, 0].flatten())
    borders.append(imgs[:, d1 - 1].flatten())
    borders.append(imgs[:, 1:d1 - 1, 0].flatten())
    borders.append(imgs[:, 1:d1 - 1, d2 - 1].flatten())
    borders = torch.cat(borders)
    if torch.median(borders) == 0.0:
        return 1.0
    return 0.0
