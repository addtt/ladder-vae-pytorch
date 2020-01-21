import torch
from torch import nn

from nn import ResidualBlock, ResidualGatedBlock
from stochastic import NormalStochasticBlock2d


class TopDownLayer(nn.Module):
    """
    Top-down layer, including stochastic sampling, KL computation, and small
    deterministic ResNet with upsampling.

    The architecture when doing inference is roughly as follows:
       p_params = output of top-down layer above
       bu = inferred bottom-up value at this layer
       q_params = merge(bu, p_params)
       z = stochastic_layer(q_params)
       possibly get skip connection from previous top-down layer
       top-down deterministic ResNet

    When doing generation only, the value bu is not available, the
    merge layer is not used, and z is sampled directly from p_params.

    If this is the top layer, at inference time, the uppermost bottom-up value
    is used directly as q_params, and p_params are defined in this layer
    (while they are usually taken from the previous layer), and can be learned.
    """
    def __init__(self, z_dim, n_res_blocks, n_filters, is_top_layer=False,
                 downsampling_steps=None, nonlin=None, merge_type=None,
                 batchnorm=True, dropout=None, stochastic_skip=False,
                 res_block_type=None, gated=None, learn_top_prior=False,
                 top_prior_param_shape=None, analytical_kl=False):

        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip
        self.learn_top_prior = learn_top_prior
        self.analytical_kl = analytical_kl

        # Define top layer prior parameters, possibly learnable
        if is_top_layer:
            self.top_prior_params = nn.Parameter(
                torch.zeros(top_prior_param_shape),
                requires_grad=learn_top_prior
            )

        # Downsampling steps left to do in this layer
        dws_left = downsampling_steps

        # Define deterministic top-down block: sequence of deterministic
        # residual blocks with downsampling when needed.
        block_list = []
        for _ in range(n_res_blocks):
            do_resample = False
            if dws_left > 0:
                do_resample = True
                dws_left -= 1
            block_list.append(
                TopDownDeterministicResBlock(
                    n_filters,
                    n_filters,
                    nonlin,
                    upsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                )
            )
        self.deterministic_block = nn.Sequential(*block_list)

        # Define stochastic block with 2d convolutions
        self.stochastic = NormalStochasticBlock2d(
            n_filters,
            z_dim,
            n_filters,
            transform_p_params=(not is_top_layer),
        )

        if not is_top_layer:

            # Merge layer, combine bottom-up inference with top-down
            # generative to give posterior parameters
            self.merge = MergeLayer(
                channels=n_filters,
                merge_type=merge_type,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
            )

            # Skip connection that goes around the stochastic top-down layer
            if stochastic_skip:
                self.skip_connection_merger = SkipConnectionMerger(
                    channels=n_filters,
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                )

    def forward(self, input_=None, skip_connection_input=None,
                inference_mode=False, bu_value=None, n_img_prior=None,
                forced_latent=None, use_mode=False,
                force_constant_output=False):

        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        # If top layer, define parameters of prior p(z_L)
        if self.is_top_layer:
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.expand(n_img_prior, -1, -1, -1)

        # Else the input from the layer above is the prior parameters
        else:
            p_params = input_

        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        if inference_mode:
            if self.is_top_layer:
                q_params = bu_value
            else:
                q_params = self.merge(bu_value, p_params)

        # In generative mode, q is not used
        else:
            q_params = None

        # Sample from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None
        x, aux = self.stochastic(
            p_params=p_params,
            q_params=q_params,
            forced_latent=forced_latent,
            use_mode=use_mode,
            force_constant_output=force_constant_output,
            analytical_kl=self.analytical_kl,
        )

        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            x = self.skip_connection_merger(x, skip_connection_input)

        # Save activation before residual block: could be the skip
        # connection input in the next layer
        x_pre_residual = x

        # Last top-down block (sequence of residual blocks)
        x = self.deterministic_block(x)

        aux_out = {
            'z': aux['z'],
            'kl': aux['kl_samplewise'],
            'kl_spatial': aux['kl_spatial'],  # (B, H, W)
            'logprob_p': aux['logprob_p'],
            'logprob_q': aux['logprob_q'],
        }
        return x, x_pre_residual, aux_out


class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer for inference, roughly the same as the
    small deterministic Resnet in top-down layers. Consists of a sequence of
    bottom-up deterministic residual blocks with downsampling.
    """

    def __init__(self, n_res_blocks, n_filters, downsampling_steps=0,
                 nonlin=None, batchnorm=True, dropout=None, res_block_type=None,
                 gated=None):
        super().__init__()

        bu_blocks = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1
            bu_blocks.append(
                BottomUpDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    nonlin=nonlin,
                    downsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                )
            )
        self.net = nn.Sequential(*bu_blocks)

    def forward(self, x):
        return self.net(x)


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling steps (each by a factor of 2).

    The mode can be top-down or bottom-up, and the block does up- and
    down-sampling by a factor of 2, respectively. Resampling is performed at
    the beginning of the block, through strided convolution.

    The number of channels is adjusted at the beginning and end of the block,
    through convolutional layers with kernel size 1. The number of internal
    channels is by default the same as the number of output channels, but
    min_inner_channels overrides this behaviour.

    Other parameters: kernel size, nonlinearity, and groups of the internal
    residual block; whether batch normalization and dropout are performed;
    whether the residual path has a gate layer at the end. There are a few
    residual block structures to choose from.
    """
    def __init__(self, mode, c_in, c_out, nonlin=nn.LeakyReLU, resample=False,
                 res_block_kernel=None, groups=1, batchnorm=True, res_block_type=None,
                 dropout=None, min_inner_channels=None, gated=None):
        super().__init__()
        assert mode in ['top-down', 'bottom-up']
        if min_inner_channels is None:
            min_inner_channels = 0
        inner_filters = max(c_out, min_inner_channels)

        # Define first conv layer to change channels and/or up/downsample
        if resample:
            if mode == 'bottom-up':  # downsample
                self.pre_conv = nn.Conv2d(
                    in_channels=c_in,
                    out_channels=inner_filters,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    groups=groups
                )
            elif mode == 'top-down':  # upsample
                self.pre_conv = nn.ConvTranspose2d(
                    in_channels=c_in,
                    out_channels=inner_filters,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    groups=groups,
                    output_padding=1
                )
        elif c_in != inner_filters:
            self.pre_conv = nn.Conv2d(c_in, inner_filters, 1, groups=groups)
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_filters,
            nonlin=nonlin,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            dropout=dropout,
            gated=gated,
            block_type=res_block_type,
        )

        # Define last conv layer to get correct num output channels
        if inner_filters != c_out:
            self.post_conv = nn.Conv2d(inner_filters, c_out, 1, groups=groups)
        else:
            self.post_conv = None

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        x = self.res(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownDeterministicResBlock(ResBlockWithResampling):
    def __init__(self, *args, upsample=False, **kwargs):
        kwargs['resample'] = upsample
        super().__init__('top-down', *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):
    def __init__(self, *args, downsample=False, **kwargs):
        kwargs['resample'] = downsample
        super().__init__('bottom-up', *args, **kwargs)


class MergeLayer(nn.Module):
    """
    Merge two 4D input tensors by concatenating along dim=1 and passing the
    result through 1) a convolutional 1x1 layer, or 2) a residual block
    """
    def __init__(self, channels, merge_type, nonlin=nn.LeakyReLU,
                 batchnorm=True, dropout=None, res_block_type=None):
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3
        assert len(channels) == 3

        if merge_type == 'linear':
            self.layer = nn.Conv2d(channels[0] + channels[1], channels[2], 1)
        elif merge_type == 'residual':
            self.layer = nn.Sequential(
                nn.Conv2d(channels[0] + channels[1], channels[2], 1, padding=0),
                ResidualGatedBlock(channels[2], nonlin, batchnorm=batchnorm,
                                   dropout=dropout,
                                   block_type=res_block_type),
            )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        return self.layer(x)


class SkipConnectionMerger(MergeLayer):
    """
    By default for now simply a merge layer.
    """

    merge_type = 'residual'

    def __init__(self, channels, nonlin, batchnorm, dropout, res_block_type):
        super().__init__(
            channels, self.merge_type, nonlin, batchnorm, dropout=dropout,
            res_block_type=res_block_type)