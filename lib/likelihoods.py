import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .stochastic import normal_rsample, logistic_rsample, sample_from_discretized_mix_logistic


class LikelihoodModule(nn.Module):

    def distr_params(self, x):
        pass

    @staticmethod
    def mean(params):
        pass

    @staticmethod
    def mode(params):
        pass

    @staticmethod
    def sample(params):
        pass

    def log_likelihood(self, x, params):
        pass

    def forward(self, input_, x):
        distr_params = self.distr_params(input_)
        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        if x is None:
            ll = None
        else:
            ll = self.log_likelihood(x, distr_params)
        dct = {
            'mean': mean,
            'mode': mode,
            'sample': sample,
            'params': distr_params,
        }
        return ll, dct


class BernoulliLikelihood(LikelihoodModule):
    def __init__(self, ch_in, color_channels):
        super().__init__()
        self.parameter_net = nn.Conv2d(
            ch_in,
            color_channels,
            kernel_size=3,
            padding=1)

    def distr_params(self, x):
        x = self.parameter_net(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def mean(params):
        return params

    @staticmethod
    def mode(params):
        return torch.round(params)

    @staticmethod
    def sample(params):
        return (torch.rand_like(params) < params).float()

    def log_likelihood(self, x, params):
        return log_bernoulli(x, params, reduce='none')


class GaussianLikelihood(LikelihoodModule):
    def __init__(self, ch_in, color_channels):
        super().__init__()
        self.parameter_net = nn.Conv2d(
            ch_in,
            2 * color_channels,
            kernel_size=3,
            padding=1)

    def distr_params(self, x):
        x = self.parameter_net(x)
        mean, lv = x.chunk(2, dim=1)
        params = {
            'mean': mean,
            'logvar': lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        sample = normal_rsample((
            params['mean'],
            params['logvar']
        ))
        return sample

    def log_likelihood(self, x, params):
        logprob = log_normal(
            x,
            params['mean'],
            params['logvar'],
            reduce='none'
        )
        return logprob



class DiscretizedLogisticLikelihood(LikelihoodModule):
    """
    Assume input data to be originally uint8 (0, ..., 255) and then rescaled
    by 1/255: discrete values in {0, 1/255, ..., 255/255}.
    If using the discretize logistic logprob implementation here, this should
    be rescaled by 255/256 and shifted by <1/256 in this class. So the data is
    inside 256 bins between 0 and 1.

    Note that mean and logscale are parameters of the underlying continuous
    logistic distribution, not of its discretization.
    """

    log_scale_bias = -1.

    def __init__(self, ch_in, color_channels, n_bins, double=False):
        super().__init__()
        self.n_bins = n_bins
        self.double_precision = double
        self.parameter_net = nn.Conv2d(
            ch_in,
            2 * color_channels,
            kernel_size=3,
            padding=1)

    def distr_params(self, x):
        x = self.parameter_net(x)
        mean, ls = x.chunk(2, dim=1)
        ls = ls + self.log_scale_bias
        ls = ls.clamp(min=-7.)
        mean = mean + 0.5   # initialize to mid interval
        params = {
            'mean': mean,
            'logscale': ls,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        # We're not quantizing 8bit, but it doesn't matter
        sample = logistic_rsample((
            params['mean'],
            params['logscale']
        ))
        sample = sample.clamp(min=0., max=1.)
        return sample

    def log_likelihood(self, x, params):
        # Input data x should be inside (not at the edge) n_bins equally-sized
        # bins between 0 and 1. E.g. if n_bins=256 the 257 bin edges are:
        # 0, 1/256, ..., 255/256, 1.

        x = x * (255/256) + 1/512

        logprob = log_discretized_logistic(
            x,
            params['mean'],
            params['logscale'],
            n_bins=self.n_bins,
            reduce='none',
            double=self.double_precision
        )
        return logprob


class DiscretizedLogisticMixLikelihood(LikelihoodModule):
    """
    Sampling and loss computation are based on the original tf code.

    Assume input data to be originally uint8 (0, ..., 255) and then rescaled
    by 1/255: discrete values in {0, 1/255, ..., 255/255}.
    When using the original discretize logistic mixture logprob implementation,
    this data should be rescaled to be in [-1, 1].

    Mean and mode are not implemented for now.

    Color channels for now is fixed to 3 and n_bins to 256.
    """

    log_scale_bias = -1.

    def __init__(self, ch_in, n_components=5):
        super().__init__()
        self.parameter_net = nn.Conv2d(
            ch_in,
            10 * n_components,
            kernel_size=3,
            padding=1)

    def distr_params(self, x):
        x = self.parameter_net(x)
        params = {
            'mean': None,  # TODO
            'all_params': x
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        sample = sample_from_discretized_mix_logistic(params['all_params'])
        sample = (sample + 1) / 2
        sample = sample.clamp(min=0., max=1.)
        return sample

    def log_likelihood(self, x, params):
        x = x * 2 - 1
        logprob = -discretized_mix_logistic_loss(x, params['all_params'])
        return logprob



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


def discretized_mix_logistic_loss(x, l):
    """
    log-likelihood for mixture of discretized logistics, assumes the data
    has been rescaled to [-1,1] interval

    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp
    """

    # channels last
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)

    # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    xs = [int(y) for y in x.size()]
    # predicted distribution, e.g. (B,32,32,100)
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + nn.Parameter(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below
    # for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999,
    # log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which
    # never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero
    # instead of selecting: this requires use to use some ugly tricks to avoid
    # potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as
    # output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (
                1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + torch.log_softmax(logit_probs, dim=-1)
    log_probs = torch.logsumexp(log_probs, dim=-1)

    # return -torch.sum(log_probs)
    loss_sep = -log_probs.sum((1, 2))  # keep batch dimension
    return loss_sep


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

    import seaborn as sns
    sns.set()

    # *** Test discretized logistic likelihood and plot examples

    # Fix predicted distribution, change true data from 0 to 1:
    # show log probability of given distribution on the range [0, 1]
    t = torch.arange(0., 1., 1 / 10000).view(-1, 1, 1, 1)
    mean_ = torch.zeros_like(t) + 0.3
    logscales = np.arange(-7., 0., 1.)
    plt.figure(figsize=(15, 8))
    for logscale_ in logscales:
        logscale = torch.tensor(logscale_).expand_as(t)
        log_prob = log_discretized_logistic(
            t, mean_, logscale, n_bins=256, reduce='none', double=True)
        plt.plot(t.flatten().numpy(), log_prob.numpy(), label='logscale={}'.format(logscale_))
    plt.xlabel('data (x)')
    plt.ylabel('logprob')
    plt.title('log DiscrLogistic(x | 0.3, scale)')
    plt.legend()
    plt.show()

    # Fix true data, change distribution:
    # show log probability of fixed data under different distributions
    logscales = np.arange(-7., 0., 1.)
    mean_ = torch.arange(0., 1., 1 / 10000).view(-1, 1, 1, 1)
    t = torch.tensor(0.3).expand_as(mean_)
    plt.figure(figsize=(15, 8))
    for logscale_ in logscales:
        logscale = torch.tensor(logscale_).expand_as(mean_)
        log_prob = log_discretized_logistic(
            t, mean_, logscale, n_bins=256, reduce='none', double=True)
        plt.plot(mean_.flatten().numpy(), log_prob.numpy(), label='logscale={}'.format(logscale_))
    plt.xlabel('mean of logistic')
    plt.ylabel('logprob')
    plt.title('log DiscrLogistic(0.3 | mean, scale)')
    plt.legend()
    plt.show()
