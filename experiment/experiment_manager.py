import os

import torch
from boilr import VIExperimentManager
from boilr.nn_init import data_dependent_init
from boilr.utils import linear_anneal
from torch import optim
from torchvision.utils import save_image

from models.lvae import LadderVAE
from utils import get_imgs_pad_value
from .config import parse_args
from .data import DatasetLoader


class LVAEExperiment(VIExperimentManager):
    """
    Experiment manager.

    Data attributes:
    - 'args': argparse.Namespace containing all config parameters. When
      initializing this object, if 'args' is not given, all config
      parameters are set based on experiment defaults and user input, using
      argparse.
    - 'run_description': string description of the run that includes a timestamp
      and can be used e.g. as folder name for logging.
    - 'model': the model.
    - 'device': torch.device that is being used
    - 'dataloaders': DataLoaders, with attributes 'train' and 'test'
    - 'optimizer': the optimizer
    """


    def make_datamanager(self):
        cuda = self.device.type == 'cuda'
        return DatasetLoader(self.args, cuda)

    def make_model(self):
        args = self.args
        model = LadderVAE(
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
        )

        # Weight initialization
        if args.simple_data_dependent_init:

            # Get batch
            t = [self.dataloaders.train.dataset[i] for i in range(args.batch_size)]
            t = torch.stack(tuple(t[i][0] for i in range(len(t))))

            # Use batch for data dependent init
            data_dependent_init(self.model, {'x': t.to(self.device)})

        return model

    def make_optimizer(self):
        args = self.args
        optimizer = optim.Adamax(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        return optimizer


    @staticmethod
    def _parse_args():
        """
        Parse command-line arguments defining experiment settings.

        :return: args: argparse.Namespace with experiment settings
        """
        return parse_args()


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



    def forward_pass(self, model, x, y=None):
        """
        Simple single-pass model evaluation. It consists of a forward pass
        and computation of all necessary losses and metrics.
        """

        # Forward pass
        x = x.to(self.device, non_blocking=True)
        model_out = model(x)
        recons_sep = -model_out['ll']
        kl_sep = model_out['kl_sep']
        kl = model_out['kl']
        kl_loss = model_out['kl_loss']

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

        output = {
            'loss': loss,
            'elbo': elbo,
            'elbo_sep': elbo_sep,
            'kl': kl,
            'l2': l2,
            'recons': recons,
            'out_mean': model_out['out_mean'],
            'out_mode': model_out['out_mode'],
            'out_sample': model_out['out_sample'],
            'likelihood_params': model_out['likelihood_params'],
        }
        if 'kl_avg_layerwise' in model_out:
            output['kl_avg_layerwise'] = model_out['kl_avg_layerwise']

        return output


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
            pad_value = get_imgs_pad_value(sample)
            fname = os.path.join(img_folder, 'sample_' + str(step) + '.png')
            save_image(sample, fname, nrow=n, pad_value=pad_value)

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
        outputs = self.forward_pass(self.model, x)
        x = x.to(self.device)
        imgs = torch.stack([
            x[:n_img],
            outputs['out_sample'][:n_img]])
        imgs = imgs.permute(1, 0, 2, 3, 4)
        imgs = imgs.reshape(n ** 2, x.size(1), x.size(2), x.size(3))
        pad_value = get_imgs_pad_value(imgs)
        save_image(imgs.cpu(), fname, nrow=n, pad_value=pad_value)
