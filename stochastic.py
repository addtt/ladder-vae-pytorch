import torch
from torch import nn
from torch.distributions.normal import Normal


class NormalStochasticBlock2d(nn.Module):
    def __init__(self, c_in, c_vars, c_out, kernel=3, transform_p_params=True):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = nn.Conv2d(c_vars, c_out, kernel, padding=pad)

    def forward(self, p_params, q_params=None, forced_latent=None,
                use_mode=False, force_constant_output=False):

        assert (forced_latent is None) or (not use_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        if q_params is not None:
            q_params = self.conv_in_q(q_params)
            mu_lv = q_params
        else:
            mu_lv = p_params

        if forced_latent is None:
            if use_mode:
                z = torch.chunk(mu_lv, 2, dim=1)[0]
            else:
                z = normal_rsample(mu_lv)
        else:
            z = forced_latent

        # Copy one sample (and distrib parameters) over the whole batch
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = p_params[0:1].expand_as(p_params).clone()

        # Output of stochastic layer
        out = self.conv_out(z)

        kl_elementwise = kl_samplewise = kl_spatial_analytical = None
        logprob_q = None

        # Compute log p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        p = Normal(p_mu, (p_lv / 2).exp())
        logprob_p = p.log_prob(z).sum((1, 2, 3))

        if q_params is not None:

            # Compute log q(z)
            q_mu, q_lv = q_params.chunk(2, dim=1)
            q = Normal(q_mu, (q_lv / 2).exp())
            logprob_q = q.log_prob(z).sum((1, 2, 3))

            # Compute KL estimate (expectation by sampling)
            kl_elementwise = kl_normal_mc(z, p_params, q_params)
            kl_samplewise = kl_elementwise.sum((1, 2, 3))

            # Compute spatial KL analytically (but conditioned on samples from
            # previous layers)
            kl_spatial_analytical = -0.5 * (1 + q_lv - q_mu.pow(2) - q_lv.exp())
            kl_spatial_analytical = kl_spatial_analytical.sum(1)

        data = {
            'z': z,
            'p_params': p_params,
            'q_params': q_params,
            'logprob_p': logprob_p,
            'logprob_q': logprob_q,
            'kl_elementwise': kl_elementwise,
            'kl_samplewise': kl_samplewise,
            'kl_spatial': kl_spatial_analytical,
        }
        return out, data


def normal_rsample(mu_lv):
    """
    Returns a sample from Normal with specified mean and log variance.
    :param mu_lv: a tensor containing mean and log variance along dim=1,
            or a tuple (mean, log variance)
    :return: a reparameterized sample with the same size as the input
            mean and log variance
    """
    try:
        mu, lv = torch.chunk(mu_lv, 2, dim=1)
    except TypeError:
        mu, lv = mu_lv
    eps = torch.randn_like(mu)
    std = (lv / 2).exp()
    return eps * std + mu


def logistic_rsample(mu_ls):
    """
    Returns a sample from Logistic with specified mean and log scale.
    :param mu_ls: a tensor containing mean and log scale along dim=1,
            or a tuple (mean, log scale)
    :return: a reparameterized sample with the same size as the input
            mean and log scale
    """

    # Get parameters
    try:
        mu, log_scale = torch.chunk(mu_ls, 2, dim=1)
    except TypeError:
        mu, log_scale = mu_ls
    scale = log_scale.exp()

    # Get uniform sample in open interval (0, 1)
    u = torch.zeros_like(mu)
    u.uniform_(1e-7, 1 - 1e-7)

    # Transform into logistic sample
    sample = mu + scale * (torch.log(u) - torch.log(1 - u))

    return sample


def sample_from_discretized_mix_logistic(l):
    """
    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp
    """

    def to_one_hot(tensor, n):
        one_hot = torch.zeros(tensor.size() + (n,))
        one_hot = one_hot.to(tensor.device)
        one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), 1.)
        return one_hot


    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = nn.Parameter(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def kl_normal_mc(z, p_mulv, q_mulv):
    """
    One-sample estimation of element-wise KL between two diagonal
    multivariate normal distributions. Any number of dimensions,
    broadcasting supported (be careful).

    :param z:
    :param p_mulv:
    :param q_mulv:
    :return:
    """
    p_mu, p_lv = torch.chunk(p_mulv, 2, dim=1)
    q_mu, q_lv = torch.chunk(q_mulv, 2, dim=1)
    p_std = (p_lv / 2).exp()
    q_std = (q_lv / 2).exp()
    p_distrib = Normal(p_mu, p_std)
    q_distrib = Normal(q_mu, q_std)
    return q_distrib.log_prob(z) - p_distrib.log_prob(z)
