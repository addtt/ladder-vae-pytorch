import argparse
import os

import numpy as np
import seaborn as sns
import torch
import torch.utils.data
import torch.utils.data
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from tqdm import tqdm

from experiment.data import DatasetLoader
from framework.base_experiment import BaseExperimentConfig
from framework.init import data_dependent_init
from framework.summarize import SummarizerCollection
from framework.utils import print_num_params, linear_anneal
from models.lvae import LadderVAE

sns.set()


class ExperimentConfig(BaseExperimentConfig):
    """
    Experiment configuration.

    Data attributes:

    - 'args': argparse.Namespace containing all config parameters. When
      initializing the ExperimentConfig, if 'args' is not given, all config
      parameters are set based on experiment defaults and user input, using
      argparse.

    - 'run_description': string description of the run that includes a timestamp
      and can be used e.g. as folder name for logging.

    - 'model'

    - 'device': torch.device that is being used

    - 'dataloaders': DatasetLoader, with attributes 'train', 'test',
        'data_shape', 'img_size', 'color_ch'

    - 'optimizer'

    """

    @staticmethod
    def _parse_args():
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


    @staticmethod
    def _make_run_description(args):
        """
        Create a string description of the run. It is used in the names of the
        logging folders.

        :param args: experiment config
        :return: the run description
        """
        s = ''
        s += args.dataset_name
        s += ',{}ly'.format(len(args.z_dims))
        # s += ',z=' + str(args.z_dims).replace(" ", "")
        # s += ',dwn=' + str(args.downsample).replace(" ", "")
        s += ',{}bpl'.format(args.blocks_per_layer)
        if args.skip_connections:
            s += ',skip'
        if args.gated:
            s += ',gated'
        s += ',block=' + args.residual_type
        if args.beta_anneal != 0:
            s += ',b{}'.format(args.beta_anneal)
        s += ',{}'.format(args.nonlin)
        if args.free_bits > 0:
            s += ',freebits={}'.format(args.free_bits)
        if args.dropout is not None:
            s += ',dropout={}'.format(args.dropout)
        if args.learn_top_prior:
            s += ',learnprior'
        if args.weight_decay > 0.0:
            s += ',wd={}'.format(args.weight_decay)
        s += ',seed{}'.format(args.seed)
        if len(args.additional_descr) > 0:
            s += ',' + args.additional_descr
        return s


    def setup(self, device, create_optimizer=True):
        """
        Set up experiment. Define data loaders, model, and optimizer.

        :param device:
        :param create_optimizer:
        """
        self.device = device

        is_cuda = device.type == 'cuda'
        print("Preparing data set...")
        self.dataloaders = DatasetLoader(self.args, is_cuda)
        print("Data shape: {}".format(self.dataloaders.data_shape))
        print("Test set size:", len(self.dataloaders.test.dataset))

        args = self.args

        # Define model
        print("Creating model...")
        self.model = LadderVAE(
            self.dataloaders.color_ch,
            z_dims=args.z_dims,
            blocks_per_layer=args.blocks_per_layer,
            downsample=args.downsample,
            merge_type=args.merge_layers,
            batchnorm=args.batch_norm,
            nonlin=args.nonlin,
            stochastic_skip=args.skip_connections,
            n_filters=args.n_filters,
            dropout=args.dropout,
            res_block_type=args.residual_type,
            free_bits=args.free_bits,
            learn_top_prior=args.learn_top_prior,
            img_shape=self.dataloaders.img_size,
            likelihood_form=args.likelihood,
            gated=args.gated,
        ).to(device)

        # Weight initialization
        if args.simple_data_dependent_init:

            # Get batch
            t = [self.dataloaders.train.dataset[i] for i in range(args.batch_size)]
            t = torch.stack(tuple(t[i][0] for i in range(len(t))))

            # Use batch for data dependent init
            data_dependent_init(self.model, {'x': t.to(device)})

        print_num_params(self.model, max_depth=3)

        # Define optimizer
        self.optimizer = None
        if create_optimizer:
            self.optimizer = optim.Adamax(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )


    def basic_model_eval(self, model, x):
        """
        Simple single-pass model evaluation. It consists of a forward pass
        and computation of all necessary losses and metrics.

        :param model:
        :param x:
        :return:
        """

        # Forward pass
        x = x.to(self.device, non_blocking=True)
        ll, aux = model(x)
        recons_sep = -ll
        kl_sep = aux['kl_sep']
        kl = aux['kl']
        kl_loss = aux['kl_loss']

        # ELBO
        elbo_sep = - (recons_sep + kl_sep)
        elbo = elbo_sep.mean()

        # Loss with beta
        beta = 1.
        if self.args.beta_anneal != 0:
            beta = linear_anneal(model.global_step, 0.0, 1.0, self.args.beta_anneal)
        recons = recons_sep.mean()
        loss = recons + kl_loss * beta

        # L2
        l2 = 0.0
        for p in model.parameters():
            l2 = l2 + torch.sum(p ** 2)
        l2 = l2.sqrt()

        metrics = {
            'loss': loss,
            'elbo': elbo,
            'elbo_sep': elbo_sep,
            'kl': kl,
            'l2': l2,
            'recons': recons,
            'out_mean': aux['out_mean'],
            'out_mode': aux['out_mode'],
            'out_sample': aux['out_sample'],
            'likelihood_params': aux['likelihood_params'],
        }
        if 'kl_avg_layerwise' in aux:
            metrics['kl_avg_layerwise'] = aux['kl_avg_layerwise']

        return metrics


    @staticmethod
    def print_train_log(step, epoch, summaries):
        s = "       [step {}]   loss: {:.5g}   ELBO: {:.5g}   recons: {:.3g}   KL: {:.3g}"
        s = s.format(
            step,
            summaries['loss/loss'],
            summaries['elbo/elbo'],
            summaries['elbo/recons'],
            summaries['elbo/kl'])
        print(s)


    @staticmethod
    def print_test_log(summaries, step=None, epoch=None):
        log_string = "       "
        if epoch is not None:
            log_string += "[step {}, epoch {}]   ".format(step, epoch)
        log_string += "ELBO {:.5g}   recons: {:.3g}   KL: {:.3g}".format(
            summaries['elbo/elbo'], summaries['elbo/recons'], summaries['elbo/kl'])
        ll_key = None
        for k in summaries.keys():
            if k.find('elbo_IW') > -1:
                ll_key = k
                iw_samples = k.split('_')[-1]
                break
        if ll_key is not None:
            log_string += "   marginal log-likelihood ({}) {:.5g}".format(
                iw_samples, summaries[ll_key])

        print(log_string)


    @staticmethod
    def get_metrics_dict(results):
        metrics_dict = {
            'loss/loss': results['loss'].item(),
            'elbo/elbo': results['elbo'].item(),
            'elbo/recons': results['recons'].item(),
            'elbo/kl': results['kl'].item(),
            'l2/l2': results['l2'].item(),
        }
        try:
            for i in range(len(results['kl_avg_layerwise'])):
                key = 'kl_layers/kl_layer_{}'.format(i)
                metrics_dict[key] = results['kl_avg_layerwise'][i].item()
        except (AttributeError, KeyError):
            pass
        return metrics_dict


    def test_procedure(self, iw_samples=None):
        """
        Execute test procedure for the experiment. This typically includes
        collecting metrics on the test set using model_simple_eval().
        For example in variational inference we might be interested in
        repeating this many times to derive the importance-weighted ELBO.

        :param iw_samples: number of samples for the importance-weighted ELBO.
                The other metrics are also averaged over all these samples,
                yielding a more accurate estimate.
        :return: summaries (dict)
        """

        # Shorthand
        test_loader = self.dataloaders.test
        step = self.model.global_step
        args = self.args
        n_test = len(test_loader.dataset)

        # If it's time to estimate log likelihood, use many samples.
        # If given, use the given number.
        if iw_samples is None:
            iw_samples = 1
            if step % args.loglik_interval == 0 and step > 0:
                iw_samples = args.loglik_samples

        # Setup
        summarizers = SummarizerCollection(mode='sum')
        progress = tqdm(total=len(test_loader) * iw_samples, desc='test ')
        all_elbo_sep = torch.zeros(n_test, iw_samples)
        for batch_idx, (x, _) in enumerate(test_loader):
            for i in range(iw_samples):
                outputs = self.basic_model_eval(self.model, x)

                # elbo_sep shape (batch size,)
                i_start = batch_idx * args.test_batch_size
                i_end = (batch_idx + 1) * args.test_batch_size
                all_elbo_sep[i_start: i_end, i] = outputs['elbo_sep'].detach()

                metrics_dict = self.get_metrics_dict(outputs)
                multiplier = (x.size(0) / n_test) / iw_samples
                for k in metrics_dict:
                    metrics_dict[k] *= multiplier
                summarizers.add(metrics_dict)

                progress.update()
        progress.close()

        if iw_samples > 1:
            # Shape (test set size,)
            elbo_iw = torch.logsumexp(all_elbo_sep, dim=1)
            elbo_iw = elbo_iw - np.log(iw_samples)

            # Mean over test set (scalar)
            elbo_iw = elbo_iw.mean().item()
            key = 'elbo/elbo_IW_{}'.format(iw_samples)
            summarizers.add({key: elbo_iw})

        summaries = summarizers.get_all(reset=True)

        return summaries


    def additional_testing(self, img_folder):
        """
        Perform additional testing, including possibly generating images.

        In this case, save samples from the generative model, and pairs
        input/reconstruction from the test set.

        :param img_folder: folder to store images
        """

        step = self.model.global_step

        if not self.args.dry_run:

            # Saved images will have n**2 sub-images
            n = 8

            # Save model samples
            sample = self.model.sample_prior(n ** 2)
            fname = os.path.join(img_folder, 'sample_' + str(step) + '.png')
            save_image(sample, fname, nrow=n)

            # Get first test batch
            (x, _) = next(iter(self.dataloaders.test))
            fname = os.path.join(img_folder, 'reconstruction_' + str(step) + '.png')

            # Save model original/reconstructions
            self.save_input_and_recons(x, fname, n)


    def save_input_and_recons(self, x, fname, n):
        n_img = n ** 2 // 2
        if x.shape[0] < n_img:
            msg = ("{} data points required, but given batch has size {}. "
                   "Please use a larger batch.".format(n_img, x.shape[0]))
            raise RuntimeError(msg)
        outputs = self.basic_model_eval(self.model, x)
        x = x.to(self.device)
        imgs = torch.stack([
            x[:n_img],
            outputs['out_sample'][:n_img]])
        imgs = imgs.permute(1, 0, 2, 3, 4)
        imgs = imgs.reshape(n ** 2, x.size(1), x.size(2), x.size(3))
        save_image(imgs.cpu(), fname, nrow=n)
