import torch
from torch import nn

from framework.utils import to_np, is_conv, is_linear

debug = False


def _get_data_dep_hook(init_scale=1.):
    def hook(module, inp, out):
        inp = inp[0]

        out_size = out.size()

        if is_conv(module):
            separation_dim = 1
        elif is_linear(module):
            separation_dim = -1
        dims = tuple([i for i in range(out.dim()) if i != separation_dim])
        mean = out.mean(dims, keepdim=True)
        var = out.var(dims, keepdim=True)

        if debug:
            print("Shapes:\n   input:  {}\n   output: {}\n   weight: {}".format(
                inp.size(), out_size, module.weight.size()))
            print("Dims to compute stats over:", dims)
            print("Input statistics:\n   mean: {}\n   var: {}".format(
                to_np(inp.mean(dims)), to_np(inp.var(dims))))
            print("Output statistics:\n   mean: {}\n   var: {}".format(
                to_np(out.mean(dims)), to_np(out.var(dims))))
            print("Weight statistics:   mean: {}   var: {}".format(
                to_np(module.weight.mean()), to_np(module.weight.var())))

        # Given channel y[i] we want to get
        #   y'[i] = (y[i] - mu[i]) * is / s[i]
        #         = (b[i] - mu[i]) * is / s[i] + sum_k (w[i, k] * is / s[i] * x[k])
        # where * is 2D convolution, k denotes input channels, mu[i] is the
        # sample mean of channel i, s[i] the sample variance, b[i] the current
        # bias, 'is' the initial scale, and w[i, k] the weight kernel for input
        # k and output i.
        # Therefore the correct bias and weights are:
        #   b'[i] = is * (b[i] - mu[i]) / s[i]
        #   w'[i, k] = w[i, k] * is / s[i]
        # And finally we can modify in place the output to get y'.

        scale = torch.sqrt(var + 1e-5)

        # Fix bias
        module.bias.data = (module.bias.data - mean.flatten()) * init_scale / scale.flatten()

        # Get correct dimension if transposed conv
        transp_conv = getattr(module, 'transposed', False)
        ch_out_dim = 1 if transp_conv else 0  # TODO handle groups

        # Fix weight
        size = tuple(-1 if i == ch_out_dim else 1 for i in range(out.dim()))
        weight_size = module.weight.size()
        module.weight.data *= init_scale / scale.view(size)
        assert module.weight.size() == weight_size

        # Fix output in-place so we can continue forward pass
        out.data -= mean
        out.data *= init_scale / scale

        assert out.size() == out_size

    return hook


def data_dependent_init(model, model_input_dict, init_scale=.1):
    hook_handles = []
    modules = filter(lambda m: is_conv(m) or is_linear(m), model.modules())
    for module in modules:
        # Init module parameters before forward pass
        nn.init.kaiming_normal_(module.weight.data)
        module.bias.data.zero_()

        # Forward hook: data-dependent initialization
        hook_handle = module.register_forward_hook(_get_data_dep_hook(init_scale))
        hook_handles.append(hook_handle)

    # Forward pass one minibatch
    model(**model_input_dict)  # dry-run

    # Remove forward hooks
    for hook_handle in hook_handles:
        hook_handle.remove()


#### Test #####

if __name__ == '__main__':
    debug = True

    # *** Test simple data-dependent init

    def do_test(x, layer):
        layer.bias.data.zero_()
        print("Output stats before:", layer(x).mean().item(), layer(x).var().item())
        handle = layer.register_forward_hook(_get_data_dep_hook(init_scale=0.5))
        y = layer(x)
        print("Output stats after:", y.mean().item(), y.var().item())
        handle.remove()


    # shape 64, 3, 5, 5
    x__ = (torch.rand(64, 3, 5, 5) - 0.2) * 20

    # Test Conv2d
    print("\n\n *** TEST Conv2d\n")
    do_test(x__, nn.Conv2d(3, 4, 3, padding=1))

    # Test ConvTranspose2d
    print("\n\n *** TEST ConvTranspose2d\n")
    do_test(x__, nn.ConvTranspose2d(3, 4, 3, padding=1))

    # Test Linear
    print("\n\n *** TEST Linear\n")
    x__ = x__.view(64, 25 * 3)  # flatten
    do_test(x__, nn.Linear(25 * 3, 8))
