import torch
from torch import nn
from torch.distributions.normal import Normal

# TODO add Categorical

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
                from_mode=False, force_constant_output=False):

        assert (forced_latent is None) or (not from_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        if q_params is not None:
            q_params = self.conv_in_q(q_params)
            mu_lv = q_params
        else:
            mu_lv = p_params
            # Debugging, reduce variance when sampling
            # mu_lv = mu_lv.clone()
            # mu_lv[:, self.c_vars:] = mu_lv[:, self.c_vars:] - 2.

        if forced_latent is None:
            if from_mode:
                z = torch.chunk(mu_lv, 2, dim=1)[0]
            else:
                z = normal_reparam_sample(mu_lv)
        else:
            z = forced_latent

        # Copy one sample (and distrib parameters) over the whole batch
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = p_params[0:1].expand_as(p_params).clone()

        # print('z', z.min().item(), z.max().item())
        out = self.conv_out(z)

        # print('out', out.min().item(), out.max().item())

        kl_elementwise = kl_samplewise = None
        if q_params is not None:
            kl_elementwise = kl_normal(z, p_params, q_params)
            kl_samplewise = kl_elementwise.sum((1, 2, 3))
        data = {
            'z': z,
            'p_params': p_params,
            'q_params': q_params,
            'kl_elementwise': kl_elementwise,
            'kl_samplewise': kl_samplewise,
        }
        return out, data


def normal_reparam_sample(mu_lv):
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


def logistic_reparam_sample(mu_ls):
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


def kl_normal(z, p_mulv, q_mulv):
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
