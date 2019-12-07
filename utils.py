import numpy as np
import torch
from torch.nn import functional as F


def linear_anneal(x, start, end, steps):
    assert x >= 0
    assert steps > 0
    assert start >= 0
    assert end >= 0
    if x > steps:
        return end
    if x < 0:
        return start
    return start + (end - start) / steps * x


def to_one_hot(tensor, n):
    one_hot = torch.zeros(tensor.size() + (n,))
    one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), 1.)
    return one_hot


def to_np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x


def softplus(x, min_thresh=0.0):
    assert min_thresh <= 0
    # return f.softplus(x - min_thresh) + min_thresh
    eps = 1e-2
    h = np.log(np.exp(-min_thresh) - 1 + eps)
    return F.softplus(x + h) + min_thresh


def get_module_device(module):
    return next(module.parameters()).device


def is_conv(m):
    return isinstance(m, torch.nn.modules.conv._ConvNd)


def is_linear(m):
    return isinstance(m, torch.nn.Linear)
