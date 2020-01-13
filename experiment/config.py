import argparse

def parse_args():
    """
    Parse command-line arguments defining experiment settings.

    :return: args: argparse.Namespace with experiment settings
    """

    def list_options(lst):
        if lst:
            return "'" + "' | '".join(lst) + "'"
        return ""

    legal_merge_layers = ['linear', 'residual']
    legal_nonlin = ['relu', 'leakyrelu', 'elu', 'selu']
    legal_resblock = ['cabdcabd', 'bacdbac', 'bacdbacd']
    legal_datasets = ['static_mnist', 'cifar10', 'celeba',
                      'multi_dsprites_binary_rgb', 'svhn']
    legal_likelihoods = ['bernoulli', 'gaussian',
                         'discr_log', 'discr_log_mix']

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)

    parser.add_argument('-d', '--dataset',
                        type=str,
                        choices=legal_datasets,
                        default='static_mnist',
                        metavar='NAME',
                        dest='dataset_name',
                        help="dataset: " + list_options(legal_datasets))

    parser.add_argument('--likelihood',
                        type=str,
                        choices=legal_likelihoods,
                        metavar='NAME',
                        dest='likelihood',
                        help="likelihood: {}; default depends on dataset".format(
                            list_options(legal_likelihoods)))

    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        dest='batch_size',
                        help='training batch size')

    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        dest='test_batch_size',
                        help='test batch size')

    parser.add_argument('--lr',
                        type=float,
                        default=3e-4,
                        metavar='LR',
                        help='learning rate')

    parser.add_argument('--zdims',
                        nargs='+',
                        type=int,
                        default=[32, 32, 32],
                        metavar='DIM',
                        dest='z_dims',
                        help='list of dimensions (number of channels) for '
                             'each stochastic layer')

    parser.add_argument('--blocks-per-layer',
                        type=int,
                        default=2,
                        metavar='N',
                        help='residual blocks between stochastic layers')

    parser.add_argument('--nfilters',
                        type=int,
                        default=64,
                        metavar='N',
                        dest='n_filters',
                        help='number of channels in all residual blocks')

    parser.add_argument('--no-bn',
                        action='store_true',
                        dest='no_batch_norm',
                        help='do not use batch normalization')

    parser.add_argument('--skip',
                        action='store_true',
                        dest='skip_connections',
                        help='skip connections in generative model')

    parser.add_argument('--gated',
                        action='store_true',
                        dest='gated',
                        help='use gated layers in residual blocks')

    parser.add_argument('--downsample',
                        nargs='+',
                        type=int,
                        default=[1, 1, 1],
                        metavar='N',
                        help='list of integers, each int is the number of downsampling'
                             ' steps (by a factor of 2) before each stochastic layer')

    parser.add_argument('--learn-top-prior',
                        action='store_true',
                        help="learn the top-layer prior")

    parser.add_argument('--residual-type',
                        type=str,
                        choices=legal_resblock,
                        default='bacdbacd',
                        metavar='TYPE',
                        help="type of residual blocks: " +
                             list_options(legal_resblock))

    parser.add_argument('--merge-layers',
                        type=str,
                        choices=legal_merge_layers,
                        default='residual',
                        metavar='TYPE',
                        help="type of merge layers: " +
                             list_options(legal_merge_layers))

    parser.add_argument('--beta-anneal',
                        type=int,
                        default=0,
                        metavar='B',
                        help='steps for annealing beta from 0 to 1')

    parser.add_argument('--data-dep-init',
                        action='store_true',
                        dest='simple_data_dependent_init',
                        help='use simple data-dependent initialization to '
                             'normalize outputs of affine layers')

    parser.add_argument('--wd',
                        type=float,
                        default=0.0,
                        dest='weight_decay',
                        help='weight decay')

    parser.add_argument('--nonlin',
                        type=str,
                        choices=legal_nonlin,
                        default='elu',
                        metavar='F',
                        help="nonlinear activation: " +
                             list_options(legal_nonlin))

    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        metavar='D',
                        help='dropout probability (in deterministic layers)')

    parser.add_argument('--freebits',
                        type=float,
                        default=0.0,
                        metavar='N',
                        dest='free_bits',
                        help='free bits (nats)')

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='N',
                        help='random seed')

    parser.add_argument('--tr-log-interv',
                        type=int,
                        default=10000,
                        metavar='N',
                        dest='log_interval',
                        help='number of batches before logging train status')

    parser.add_argument('--ts-log-interv',
                        type=int,
                        default=10000,
                        metavar='N',
                        dest='test_log_interval',
                        help='number of batches before logging test status')

    parser.add_argument('--ll-interv',
                        type=int,
                        default=50000,
                        metavar='N',
                        dest='loglik_interval',
                        help='number of batches before evaluating log likelihood')

    parser.add_argument('--ll-samples',
                        type=int,
                        default=100,
                        metavar='N',
                        dest='loglik_samples',
                        help='number of importance samples to evaluate log likelihood')

    parser.add_argument('--ckpt-interv',
                        type=int,
                        default=100000,
                        metavar='N',
                        dest='checkpoint_interval',
                        help='number of batches before saving model checkpoint')

    parser.add_argument('--nocuda',
                        action='store_true',
                        dest='no_cuda',
                        help='do not use cuda')

    parser.add_argument('--descr',
                        type=str,
                        default='',
                        metavar='STR',
                        dest='additional_descr',
                        help='additional description for experiment name')

    parser.add_argument('--dry-run',
                        action='store_true',
                        dest='dry_run',
                        help='do not save anything to disk')

    args = parser.parse_args()

    if len(args.z_dims) != len(args.downsample):
        msg = (
            "length of list of latent dimensions ({}) does not match "
            "length of list of downsampling factors ({})").format(
            len(args.z_dims), len(args.downsample))
        raise RuntimeError(msg)

    assert args.weight_decay >= 0.0
    assert 0.0 <= args.dropout <= 1.0
    if args.dropout < 1e-5:
        args.dropout = None
    assert args.free_bits >= 0.0
    assert args.loglik_interval % args.test_log_interval == 0
    args.batch_norm = not args.no_batch_norm

    likelihood_map = {
        'static_mnist': 'bernoulli',
        'multi_dsprites_binary_rgb': 'bernoulli',
        'cifar10': 'discr_log_mix',
        'celeba': 'discr_log_mix',
        'svhn': 'discr_log_mix',
    }
    if args.likelihood is None:  # default
        args.likelihood = likelihood_map[args.dataset_name]

    return args
