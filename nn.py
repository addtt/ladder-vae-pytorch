import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.

    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (batchnorm, activation, conv, dropout).
    """

    default_kernel_size = (3, 3)

    def __init__(self, channels, nonlin, kernel=None, groups=1,
                 batchnorm=True, block_type=None, dropout=None, gated=None):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        pad = [k // 2 for k in kernel]
        self.gated = gated

        modules = []

        if block_type == 'cabdcabd':
            for i in range(2):
                conv = nn.Conv2d(channels, channels, kernel[i],
                                 padding=pad[i], groups=groups)
                modules.append(conv)
                modules.append(nonlin())
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                if dropout is not None:
                    modules.append(nn.Dropout2d(dropout))

        elif block_type == 'bacdbac':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin())
                conv = nn.Conv2d(channels, channels, kernel[i],
                                 padding=pad[i], groups=groups)
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(nn.Dropout2d(dropout))


        elif block_type == 'bacdbacd':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin())
                conv = nn.Conv2d(channels, channels, kernel[i],
                                 padding=pad[i], groups=groups)
                modules.append(conv)
                modules.append(nn.Dropout2d(dropout))

        else:
            raise ValueError("unrecognized block type '{}'".format(block_type))

        if gated:
            modules.append(GateLayer2d(channels, 1, nonlin))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x) + x


class ResidualGatedBlock(ResidualBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gated=True)


class GateLayer2d(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size, nonlin=nn.LeakyReLU):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nonlin()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)   # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate





#### Simple wrappers #####

class Interpolate(nn.Module):
    def __init__(self, size=None, scale=None, mode='bilinear', align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners
        )
        return out


class CropImage(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return crop_img_tensor(x, self.size)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_pass = True

    def forward(self, x):
        if self.first_pass:
            print(" > shape:", x.shape)
            self.first_pass = False
        return x


class Identity(nn.Module):
    def forward(self, x):
        return x


def _pad_crop_img(x, size, mode):
    """
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: tuple (height, width)
    :param mode: string ('pad' | 'crop')
    :return: padded image
    """
    assert x.dim() == 4 and len(size) == 2
    size = tuple(size)
    x_size = x.size()[2:4]
    if mode == 'pad':
        cond = x_size[0] > size[0] or x_size[1] > size[1]
    elif mode == 'crop':
        cond = x_size[0] < size[0] or x_size[1] < size[1]
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(
            mode, x_size, size))
    dr, dc = (abs(x_size[0] - size[0]), abs(x_size[1] - size[1]))
    dr1, dr2 = dr // 2, dr - (dr // 2)
    dc1, dc2 = dc // 2, dc - (dc // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dc1, dc2, dr1, dr2, 0, 0, 0, 0])
    elif mode == 'crop':
        return x[:, :, dr1:x_size[0] - dr2, dc1:x_size[1] - dc2]


def pad_img_tensor(x, size):
    """
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: iterable (height, width)
    :return: padded image
    """
    return _pad_crop_img(x, size, 'pad')


def crop_img_tensor(x, size):
    """
    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: iterable (height, width)
    :return: cropped image
    """
    return _pad_crop_img(x, size, 'crop')
