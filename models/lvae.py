import numpy as np
import torch
from torch import nn

from likelihoods import (
    BernoulliLikelihood,
    GaussianLikelihood,
    DiscretizedLogisticLikelihood,
    DiscretizedLogisticMixLikelihood)
from models.base import BaseModel
from models.lvae_layers import (
    TopDownLayer,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    BottomUpDeterministicResBlock)
from nn import crop_img_tensor, pad_img_tensor, Interpolate


class LadderVAE(BaseModel):
    def __init__(self, color_ch, z_dims, blocks_per_layer=2, downsample=None,
                 nonlin='elu', merge_type=None, batchnorm=True,
                 stochastic_skip=False, n_filters=32, dropout=None,
                 free_bits=0.0, learn_top_prior=False, likelihood=None,
                 img_shape=None, likelihood_form=None, res_block_type=None,
                 gated=False):
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
        self.likelihood = likelihood
        self.img_shape = tuple(img_shape)
        self.res_block_type = res_block_type
        self.gated = gated

        # Default: no downsampling (except for initial bottom-up block)
        if self.downsample is None:
            self.downsample = [0] * self.n_layers

        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = 2 * np.power(2, sum(self.downsample))

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
        self.first_bottom_up = nn.Sequential(
            nn.Conv2d(color_ch, n_filters, 5, padding=2, stride=2),
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
                )
            )

        # Final top-down layer
        modules = [Interpolate(scale=2)]
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
                n_filters, color_ch, 256, double=False)  # TODO double?
        elif likelihood_form == 'discr_log_mix':
            self.likelihood = DiscretizedLogisticMixLikelihood(
                n_filters, n_components=10
            )
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
        out, z, kl, logp = self.topdown_pass(bu_values)

        # Restore original image size
        out = crop_img_tensor(out, img_size)

        # Log likelihood and other info (per data point)
        ll, likelihood_info = self.likelihood(out, x)

        # kl[i] for each i has length batch_size
        # resulting kl shape: (batch_size, layers)
        kl = torch.cat([kl_layer.unsqueeze(1) for kl_layer in kl], dim=1)

        kl_sep = kl.sum(1)
        kl_avg_layerwise = kl.mean(0)
        kl_loss = self.get_free_bits_kl(kl).sum(1).mean(0)
        kl = kl_sep.mean()

        data = {
            'll': ll,
            'z': z,
            'kl': kl,
            'kl_sep': kl_sep,
            'kl_avg_layerwise': kl_avg_layerwise,
            'kl_loss': kl_loss,
            'logp': logp,
            'out_mean': likelihood_info['mean'],
            'out_mode': likelihood_info['mode'],
            'out_sample': likelihood_info['sample'],
            'likelihood_params': likelihood_info['params']
        }

        return data


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

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # log p(z) where z is the sample in the topdown pass
        logprob_p = 0.

        # Top-down inference/generation loop
        out = out_pre_residual = None
        for i in reversed(range(self.n_layers)):

            # if out is not None:
            #     print('LAYER', i, 'input', out.min().item(), out.max().item())

            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers

            # if i < self.n_layers - 1:
            #     n_img_prior = None

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
            z[i] = aux['z']
            kl[i] = aux['kl']
            logprob_p += aux['logprob_p'].mean()  # mean over batch

        # Final top-down layer
        out = self.final_top_down(out)

        return out, z, kl, logprob_p


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
        out, z, _, logp = self.topdown_pass(
            n_img_prior=n_imgs,
            mode_layers=mode_layers,
            constant_layers=constant_layers
        )
        out = crop_img_tensor(out, self.img_shape)

        # Log likelihood and other info (per data point)
        _, likelihood_data = self.likelihood(out, None)

        return likelihood_data['sample']

    # TODO quick prototype, bad code
    def new_sample_prior(self, n_imgs,
                         constant_layers=None, optimized_layers=None,
                         mode_layers=None, init_attempts=20,
                         gradient_steps=0, lr=3e-3,
                         ):

        best_log_p = -1e10
        best_z = None

        # Generate from prior
        for i in range(init_attempts):
            out, z, _, logp = self.topdown_pass(
                n_img_prior=n_imgs,
                mode_layers=[],
                constant_layers=constant_layers
            )
            if logp > best_log_p:
                best_log_p = logp
                best_z = z
        logp = best_log_p
        z = best_z

        print("init logp:", logp.item())

        params = []
        for i in optimized_layers:
            z[i].requires_grad_(True)
            params.append(z[i])

        if len(params) > 0:
            opt = torch.optim.Adam(params, lr=lr)
            with torch.enable_grad():
                for i in range(gradient_steps):
                    out, _, _, logp = self.topdown_pass(
                        n_img_prior=n_imgs,
                        # layers_from_mode=[],
                        # constant_layers=constant_layers,
                        forced_latent=z
                    )
                    loss = -logp

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    if (i+1) % 500 == 0:
                        print('  log p', logp.item())

        print('  log p(z) =', logp.item())

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

    def get_free_bits_kl(self, kl):
        # Input shape: (batch size, layers)
        # For now we have layerwise bits for each sample, NOT avg over minibatch
        if self.free_bits > 1e-4:
            return kl.clamp(min=self.free_bits)
        return kl
