import argparse
import os
import pickle
from os import path

import torch
import torch.utils.data
from boilr.utils import set_rnd_seed, get_date_str, img_grid_pad_value
from torchvision.utils import save_image

from experiment.experiment_manager import LVAEExperiment

# TODO this file is not up to date with the new boilr

default_run = ""

def main():
    eval_args = parse_args()

    set_rnd_seed(eval_args.seed)
    use_cuda = not eval_args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    date_str = get_date_str()
    print('device: {}, start time: {}'.format(device, date_str))

    # Get path to load model
    checkpoint_folder = path.join('checkpoints', eval_args.load)

    # Add date string and create folder on evaluation_results
    result_folder = path.join('evaluation_results', date_str + '_' + eval_args.load)
    img_folder = os.path.join(result_folder, 'imgs')
    os.makedirs(result_folder)
    os.makedirs(img_folder)

    # Load config
    config_path = path.join(checkpoint_folder, 'config.pkl')
    with open(config_path, 'rb') as file:
        args = pickle.load(file)

    # Modify config for testing
    args.test_batch_size = eval_args.test_batch_size
    args.dry_run = False

    experiment = LVAEExperiment(args=args)

    experiment.setup(device, create_optimizer=False)
    model = experiment.model

    # Load weights
    model.load(checkpoint_folder, device, step=eval_args.load_step)

    with torch.no_grad():
        model.eval()
        n = 12

        # Inspect representations learned by each layer
        if eval_args.inspect_layer_repr:
            inspect_layer_repr(model, img_folder, n=8)

        # Prior samples
        for i in range(eval_args.prior_samples):
            sample = model.sample_prior(n ** 2)
            pad_value = img_grid_pad_value(sample)
            fname = os.path.join(img_folder, 'sample_' + str(i) + '.png')
            save_image(sample, fname, nrow=n, pad_value=pad_value)

        fname = os.path.join(img_folder, 'reconstruction.png')
        (x, _) = next(iter(experiment.dataloaders.test))
        experiment.save_input_and_recons(x, fname, n)

        # Test procedure (with specified number of iw samples)
        summaries = experiment.test_procedure(iw_samples=eval_args.iw_samples)
        experiment.print_test_log(summaries)



def inspect_layer_repr(model, img_folder, n=8, mode=2):
    for i in range(model.n_layers):

        # if i not in [0, 3, 6, 10, 13, 16, 19]:
        #     continue

        print('layer', i)

        mode_layers = range(i)
        constant_layers = range(i + 1, model.n_layers)

        # Sample top layers once, then take many samples of a middle layer,
        # then optimize all downstream z's to maximize p(z) if gradient_steps>0
        if mode == 1:
            sample = []
            for r in range(n):
                sample.append(
                    model.new_sample_prior(
                        n,
                        constant_layers=constant_layers,
                        optimized_layers=mode_layers,
                        gradient_steps=0))
            sample = torch.cat(sample)
            pad_value = img_grid_pad_value(sample)
            fname = os.path.join(img_folder, 'sample_mode_layer' + str(i) + '.png')
            save_image(sample, fname, nrow=n, pad_value=pad_value)

        # Sample top layers once, then take many samples of a middle layer,
        # then sample from the mode in all downstream layers.
        elif mode == 2:
            sample = []
            for r in range(n):
                sample.append(
                    model.sample_prior(
                        n,
                        mode_layers=mode_layers,
                        constant_layers=constant_layers))
            sample = torch.cat(sample)
            pad_value = img_grid_pad_value(sample)
            fname = os.path.join(img_folder, 'sample_mode_layer' + str(i) + '.png')
            save_image(sample, fname, nrow=n, pad_value=pad_value)


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load',
                        type=str,
                        metavar='NAME',
                        default=default_run,
                        help="name of the run to be loaded")
    parser.add_argument('--ll',
                        action='store_true',
                        help="estimate log likelihood")
    parser.add_argument('--nll',
                        type=int,
                        default=1000,
                        dest='iw_samples',
                        metavar='N',
                        help="number of samples for log likelihood estimation")
    parser.add_argument('--ps',
                        type=int,
                        default=1,
                        dest='prior_samples',
                        metavar='N',
                        help="number of batches of samples from prior")
    parser.add_argument('--layer-repr',
                        action='store_true',
                        dest='inspect_layer_repr',
                        help='inspect layer representations. Generate samples '
                             'by sampling top layers once, then taking many '
                             'samples from a middle layer, and finally sample '
                             'the downstream layers from the conditional mode. '
                             'Do this for every layer.')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=2000,
                        dest='test_batch_size',
                        metavar='N',
                        help='test batch size')
    parser.add_argument('--load-step',
                        type=int,
                        dest='load_step',
                        metavar='N',
                        help='step of checkpoint to be loaded (default: last'
                             'available)')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='S',
                        help='random seed')
    parser.add_argument('--nocuda',
                        action='store_true',
                        dest='no_cuda',
                        help='do not use cuda')

    args = parser.parse_args()
    if not args.ll:
        args.iw_samples = 1
    return args


if __name__ == "__main__":
    main()
