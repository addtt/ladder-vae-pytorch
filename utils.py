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
