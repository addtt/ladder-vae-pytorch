import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn

import losses
from discretized_mix_logistic import (discretized_mix_logistic_loss,
                                      sample_from_discretized_mix_logistic)
from stochastic import normal_reparam_sample, logistic_reparam_sample

sns.set()

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
        return losses.log_bernoulli(x, params, reduce='none')


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
        sample = normal_reparam_sample((
            params['mean'],
            params['logvar']
        ))
        return sample

    def log_likelihood(self, x, params):
        logprob = losses.log_normal(
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
        sample = logistic_reparam_sample((
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

        logprob = losses.log_discretized_logistic(
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


if __name__ == '__main__':

    # *** Test discretized logistic likelihood and plot examples

    # Fix predicted distribution, change true data from 0 to 1:
    # show log probability of given distribution on the range [0, 1]
    t = torch.arange(0., 1., 1 / 10000).view(-1, 1, 1, 1)
    mean_ = torch.zeros_like(t) + 0.3
    logscales = np.arange(-7., 0., 1.)
    plt.figure(figsize=(15, 8))
    for logscale_ in logscales:
        logscale = torch.tensor(logscale_).expand_as(t)
        log_prob = losses.log_discretized_logistic(
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
        log_prob = losses.log_discretized_logistic(
            t, mean_, logscale, n_bins=256, reduce='none', double=True)
        plt.plot(mean_.flatten().numpy(), log_prob.numpy(), label='logscale={}'.format(logscale_))
    plt.xlabel('mean of logistic')
    plt.ylabel('logprob')
    plt.title('log DiscrLogistic(0.3 | mean, scale)')
    plt.legend()
    plt.show()
