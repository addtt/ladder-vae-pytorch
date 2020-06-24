"""
Standalone script for a couple of simple evaluations/tests of trained models.
"""

import argparse
import os
import pickle
import warnings
from os import path

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from boilr.eval import BaseOfflineEvaluator
from boilr.utils import set_rnd_seed, get_date_str
from boilr.utils.viz import img_grid_pad_value
from torchvision.utils import save_image

from experiment.experiment_manager import LVAEExperiment


class Evaluator(BaseOfflineEvaluator):

    def run(self):

        torch.set_grad_enabled(False)

        n = 12

        e = self._experiment
        e.model.eval()

        # Run evaluation and print results
        results = e.test_procedure(iw_samples=self.args.ll_samples)
        print("Eval results:\n{}".format(results))

        # Save samples
        for i in range(self.args.prior_samples):
            fname = os.path.join(self._img_folder, "samples_{}.png".format(i))
            e.generate_and_save_samples(fname, nrows=n)

        # Save input and reconstructions
        x, y = next(iter(e.dataloaders.test))
        fname = os.path.join(self._img_folder, "reconstructions.png")
        e.generate_and_save_reconstructions(x, fname, nrows=n)

        # Inspect representations learned by each layer
        if self.args.inspect_layer_repr:
            inspect_layer_repr(e.model, self._img_folder, n=n)

    # @classmethod
    # def _define_args_defaults(cls) -> dict:
    #     defaults = super(Evaluator, cls)._define_args_defaults()
    #     return defaults

    def _add_args(self, parser: argparse.ArgumentParser) -> None:

        super(Evaluator, self)._add_args(parser)

        parser.add_argument('--ll',
                            action='store_true',
                            help="estimate log likelihood with importance-"
                            "weighted bound")
        parser.add_argument('--ll-samples',
                            type=int,
                            default=100,
                            dest='ll_samples',
                            metavar='N',
                            help="number of importance-weighted samples for "
                            "log likelihood estimation")
        parser.add_argument('--ps',
                            type=int,
                            default=1,
                            dest='prior_samples',
                            metavar='N',
                            help="number of batches of samples from prior")
        parser.add_argument(
            '--layer-repr',
            action='store_true',
            dest='inspect_layer_repr',
            help='inspect layer representations. Generate samples '
            'by sampling top layers once, then taking many '
            'samples from a middle layer, and finally sample '
            'the downstream layers from the conditional mode. '
            'Do this for every layer.')

    @classmethod
    def _check_args(cls, args: argparse.Namespace) -> argparse.Namespace:
        args = super(Evaluator, cls)._check_args(args)

        if not args.ll:
            args.ll_samples = 1
        if args.load_step is not None:
            warnings.warn(
                "Loading weights from specific training step is not supported "
                "for now. The model will be loaded from the last checkpoint.")
        return args


def main():
    evaluator = Evaluator(experiment_class=LVAEExperiment)
    evaluator()


def inspect_layer_repr(model, img_folder, n=8):
    for i in range(model.n_layers):

        # print('layer', i)

        mode_layers = range(i)
        constant_layers = range(i + 1, model.n_layers)

        # Sample top layers once, then take many samples of a middle layer,
        # then sample from the mode in all downstream layers.
        sample = []
        for r in range(n):
            sample.append(
                model.sample_prior(n,
                                   mode_layers=mode_layers,
                                   constant_layers=constant_layers))
        sample = torch.cat(sample)
        pad_value = img_grid_pad_value(sample)
        fname = os.path.join(img_folder, 'sample_mode_layer' + str(i) + '.png')
        save_image(sample, fname, nrow=n, pad_value=pad_value)


if __name__ == "__main__":
    main()
