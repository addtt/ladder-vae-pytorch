import numpy as np
import torch
from boilr.models import BaseGenerativeModel
from boilr.nn import crop_img_tensor, pad_img_tensor, Interpolate, free_bits_kl
from torch import nn

from lib.likelihoods import (
    BernoulliLikelihood,
    GaussianLikelihood,
    DiscretizedLogisticLikelihood,
    DiscretizedLogisticMixLikelihood)
from .lvae_layers import (
    TopDownLayer,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    BottomUpDeterministicResBlock)


class LadderVAE(BaseGenerativeModel):
    def __init__(self, color_ch, z_dims, blocks_per_layer=2, downsample=None,
                 nonlin='elu', merge_type=None, batchnorm=True,
                 stochastic_skip=False, n_filters=32, dropout=None,
                 free_bits=0.0, learn_top_prior=False, img_shape=None,
                 likelihood_form=None, res_block_type=None, gated=False,
                 no_initial_downscaling=False, analytical_kl=False):
        super().__init__()
        self.color_ch = color_ch
        self.z_dims = z_dims
        self.blocks_per_layer = blocks_per_layer
        self.downsample = downsample
        self.n_layers = len(self.z_dims)
        self.stochastic_skip = stochastic_skip
        self.n_filters = n_filters
        self.dropout = dropout
        self.free_bits = free_bits
        self.learn_top_prior = learn_top_prior
        self.img_shape = tuple(img_shape)
        self.res_block_type = res_block_type
        self.gated = gated

        # Default: no downsampling (except for initial bottom-up block)
        if self.downsample is None:
            self.downsample = [0] * self.n_layers

        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not no_initial_downscaling:   # by default do another downscaling
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.blocks_per_layer
        assert len(self.downsample) == self.n_layers

        # Get class of nonlinear activation from string description
        nonlin = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
        }[nonlin]

        # First bottom-up layer: change num channels + downsample by factor 2
        # unless we want to prevent this
        stride = 1 if no_initial_downscaling else 2
        self.first_bottom_up = nn.Sequential(
            nn.Conv2d(color_ch, n_filters, 5, padding=2, stride=stride),
            nonlin(),
            BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
            )
        )

        # Init lists of layers
        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])

        for i in range(self.n_layers):

            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                )
            )

            # Add top-down stochastic layer at level i.
            # The architecture when doing inference is roughly as follows:
            #    p_params = output of top-down layer above
            #    bu = inferred bottom-up value at this layer
            #    q_params = merge(bu, p_params)
            #    z = stochastic_layer(q_params)
            #    possibly get skip connection from previous top-down layer
            #    top-down deterministic ResNet
            #
            # When doing generation only, the value bu is not available, the
            # merge layer is not used, and z is sampled directly from p_params.
            #
            self.top_down_layers.append(
                TopDownLayer(
                    z_dim=z_dims[i],
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=downsample[i],
                    nonlin=nonlin,
                    merge_type=merge_type,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    stochastic_skip=stochastic_skip,
                    learn_top_prior=learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=res_block_type,
                    gated=gated,
                    analytical_kl=analytical_kl,
                )
            )

        # Final top-down layer
        modules = list()
        if not no_initial_downscaling:
            modules.append(Interpolate(scale=2))
        for i in range(blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                )
            )
        self.final_top_down = nn.Sequential(*modules)

        # Define likelihood
        if likelihood_form == 'bernoulli':
            self.likelihood = BernoulliLikelihood(n_filters, color_ch)
        elif likelihood_form == 'gaussian':
            self.likelihood = GaussianLikelihood(n_filters, color_ch)
        elif likelihood_form == 'discr_log':
            self.likelihood = DiscretizedLogisticLikelihood(
                n_filters, color_ch, 256)
        elif likelihood_form == 'discr_log_mix':
            self.likelihood = DiscretizedLogisticMixLikelihood(n_filters)
        else:
            msg = "Unrecognized likelihood '{}'".format(likelihood_form)
            raise RuntimeError(msg)


    def forward(self, x):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)

        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values)

        # Restore original image size
        out = crop_img_tensor(out, img_size)

        # Log likelihood and other info (per data point)
        ll, likelihood_info = self.likelihood(out, x)

        # kl[i] for each i has length batch_size
        # resulting kl shape: (batch_size, layers)
        kl = torch.cat([kl_layer.unsqueeze(1) for kl_layer in td_data['kl']], dim=1)

        kl_sep = kl.sum(1)
        kl_avg_layerwise = kl.mean(0)
        kl_loss = free_bits_kl(kl, self.free_bits).sum()  # sum over layers
        kl = kl_sep.mean()

        output = {
            'll': ll,
            'z': td_data['z'],
            'kl': kl,
            'kl_sep': kl_sep,
            'kl_avg_layerwise': kl_avg_layerwise,
            'kl_spatial': td_data['kl_spatial'],
            'kl_loss': kl_loss,
            'logp': td_data['logprob_p'],
            'out_mean': likelihood_info['mean'],
            'out_mode': likelihood_info['mode'],
            'out_sample': likelihood_info['sample'],
            'likelihood_params': likelihood_info['params']
        }
        return output


    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values


    def topdown_pass(self, bu_values=None, n_img_prior=None,
                     mode_layers=None, constant_layers=None,
                     forced_latent=None):

        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(mode_layers) > 0 or len(constant_layers) > 0

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = ("Number of images for top-down generation has to be given "
                   "if and only if we're not doing inference")
            raise RuntimeError(msg)
        if inference_mode and prior_experiment:
            msg = ("Prior experiments (e.g. sampling from mode) are not"
                   " compatible with inference mode")
            raise RuntimeError(msg)

        # Sampled latent variables at each layer
        z = [None] * self.n_layers

        # KL divergence of each layer
        kl = [None] * self.n_layers

        # Spatial map of KL divergence for each layer
        kl_spatial = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # log p(z) where z is the sample in the topdown pass
        logprob_p = 0.

        # Top-down inference/generation loop
        out = out_pre_residual = None
        for i in reversed(range(self.n_layers)):

            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out, out_pre_residual, aux = self.top_down_layers[i](
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
            )
            z[i] = aux['z']  # sampled variable at this layer (batch, ch, h, w)
            kl[i] = aux['kl_samplewise']   # (batch, )
            kl_spatial[i] = aux['kl_spatial']  # (batch, h, w)
            logprob_p += aux['logprob_p'].mean()  # mean over batch

        # Final top-down layer
        out = self.final_top_down(out)

        data = {
            'z': z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            'kl': kl,  # list of tensors with shape (batch, )
            'kl_spatial': kl_spatial,  # list of tensors w shape (batch, h[i], w[i])
            'logprob_p': logprob_p,  # scalar, mean over batch
        }
        return out, data


    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size())
        x = pad_img_tensor(x, size)
        return x


    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, w) or (H, W)
        :return: 2-tuple (H, W)
        """

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Make size argument into (heigth, width)
        if len(size) == 4:
            size = size[2:]
        if len(size) != 2:
            msg = ("input size must be either (N, C, H, W) or (H, W), but it "
                   "has length {} (size={})".format(len(size), size))
            raise RuntimeError(msg)

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

        return padded_size


    def sample_prior(self, n_imgs, mode_layers=None, constant_layers=None):

        # Generate from prior
        out, _ = self.topdown_pass(
            n_img_prior=n_imgs,
            mode_layers=mode_layers,
            constant_layers=constant_layers
        )
        out = crop_img_tensor(out, self.img_shape)

        # Log likelihood and other info (per data point)
        _, likelihood_data = self.likelihood(out, None)

        return likelihood_data['sample']


    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        c = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, c, h, w)
        return top_layer_shape
