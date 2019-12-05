import math

import torch
from torch.nn import functional as F


def log_bernoulli(x, mean, reduce='mean'):
    log_prob = -F.binary_cross_entropy(mean, x, reduction='none')
    log_prob = log_prob.sum((1, 2, 3))
    return _reduce(log_prob, reduce)


def log_normal(x, mean, logvar, reduce='mean'):
    """
    Log of the probability density of the values x untder the Normal
    distribution with parameters mean and logvar. The sum is taken over all
    dimensions except for the first one (assumed to be batch). Reduction
    is applied at the end.
    :param x: tensor of points, with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param logvar: tensor with log-variance of distribution, shape has to be
                   either scalar or broadcastable
    :param reduce: reduction over batch: 'mean' | 'sum' | 'none'
    :return:
    """

    logvar = _input_check(x, mean, logvar, reduce)
    var = torch.exp(logvar)
    log_prob = -0.5 * (((x - mean) ** 2) / var + logvar
                       + torch.tensor(2 * math.pi).log())
    log_prob = log_prob.sum((1, 2, 3))
    return _reduce(log_prob, reduce)


def log_discretized_logistic(x, mean, log_scale, n_bins=256,
                             reduce='mean', double=False):
    """
    Log of the probability mass of the values x under the logistic distribution
    with parameters mean and scale. The sum is taken over all dimensions except
    for the first one (assumed to be batch). Reduction is applied at the end.

    Assume input data to be inside (not at the edge) of n_bins equally-sized
    bins between 0 and 1. E.g. if n_bins=256 the 257 bin edges are:
    0, 1/256, ..., 255/256, 1.
    If values are at the left edge it's also ok, but let's be on the safe side

    Variance of logistic distribution is
        var = scale^2 * pi^2 / 3

    :param x: tensor with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param log_scale: tensor with log scale of distribution, shape has to be either
                  scalar or broadcastable
    :param n_bins: bin size (default: 256)
    :param reduce: reduction over batch: 'mean' | 'sum' | 'none'
    :param double: whether double precision should be used for computations
    :return:
    """
    log_scale = _input_check(x, mean, log_scale, reduce)
    if double:
        log_scale = log_scale.double()
        x = x.double()
        mean = mean.double()
        eps = 1e-14
    else:
        eps = 1e-7
    # scale = np.sqrt(3) * torch.exp(logvar / 2) / np.pi
    scale = log_scale.exp()

    # Set values to the left of each bin
    x = torch.floor(x * n_bins) / n_bins

    cdf_plus = torch.ones_like(x)
    idx = x < (n_bins - 1) / n_bins
    cdf_plus[idx] = torch.sigmoid((x[idx] + 1 / n_bins - mean[idx]) / scale[idx])
    cdf_minus = torch.zeros_like(x)
    idx = x >= 1 / n_bins
    cdf_minus[idx] = torch.sigmoid((x[idx] - mean[idx]) / scale[idx])
    log_prob = torch.log(cdf_plus - cdf_minus + eps)
    log_prob = log_prob.sum((1, 2, 3))
    log_prob = _reduce(log_prob, reduce)
    if double:
        log_prob = log_prob.float()
    return log_prob


def _reduce(x, reduce):
    if reduce == 'mean':
        x = x.mean()
    elif reduce == 'sum':
        x = x.sum()
    return x


def _input_check(x, mean, scale_param, reduce):
    assert x.dim() == 4
    assert x.size() == mean.size()
    if scale_param.numel() == 1:
        scale_param = scale_param.view(1, 1, 1, 1)
    if reduce not in ['mean', 'sum', 'none']:
        msg = "unrecognized reduction method '{}'".format(reduce)
        raise RuntimeError(msg)
    return scale_param

if __name__ == '__main__':
    x = torch.rand(10, 3, 10, 10)
    mean = x
    logscale = torch.ones_like(x) * (-5)
    print("Loss is {:.3g}".format(log_discretized_logistic(x, mean, logscale, n_bins=5)))
    x = torch.rand(10, 3, 10, 10)
    mean = x
    logscale = torch.ones_like(x) * (-12)
    print("Loss is {:.3g}".format(log_discretized_logistic(x, mean, logscale, n_bins=5)))
    x = torch.rand(10, 3, 10, 10)
    mean = x
    logscale = torch.ones_like(x) * (-12)
    print("Loss is {:.3g}".format(log_discretized_logistic(x, mean, logscale)))
