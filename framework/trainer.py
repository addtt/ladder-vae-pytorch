import os
import pickle

import torch
import torch.utils.data
from torch.utils import tensorboard
from tqdm import tqdm

from summarize import History, SummarizerCollection
from utils import set_rnd_seed, get_date_str


class Trainer:
    """
    Generic tool for training models. All model- and experiment-specific work is
    defined and performed in an experiment object, which is provided at init.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        args = experiment.args
        assert args.checkpoint_interval % args.test_log_interval == 0

        # To save training and test metrics
        self.train_history = History()
        self.test_history = History()

        # Random seed
        set_rnd_seed(args.seed)
        
        # Pick device (cpu/cuda)
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        # Get starting time and date for logging
        date_str = get_date_str()
        
        # Print info
        print('device: {}, start time: {}'.format(self.device, date_str))

        # Create folders for logs and results, save config
        folder_str = date_str + '_' + experiment.run_description
        result_folder = os.path.join('results', folder_str)
        self.img_folder = os.path.join(result_folder, 'imgs')
        self.checkpoint_folder = os.path.join('checkpoints', folder_str)
        self.log_path = os.path.join(result_folder, 'log.pkl')
        tboard_folder = os.path.join('tensorboard_logs', folder_str)
        self.tb_writer = None
        if not args.dry_run:
            os.makedirs(result_folder)
            os.makedirs(self.img_folder)
            os.makedirs(self.checkpoint_folder)
            os.makedirs(tboard_folder)
            self.tb_writer = tensorboard.SummaryWriter(tboard_folder)
            config_path = os.path.join(self.checkpoint_folder, 'config.pkl')
            with open(config_path, 'wb') as fd:
                pickle.dump(args, fd)

        # Load dataset, create model and optimizer (kept in 'experiment' obj)
        experiment.setup(self.device)


    def run(self):

        # Setup
        e = self.experiment
        train_loader = e.dataloaders.train
        train_summarizers = SummarizerCollection(
            mode='moving_average', ma_length=1000)
        progress = None

        # Training mode
        e.model.train()

        # Main loop
        for epoch in range(1, e.max_epochs + 1):
            for batch_idx, (x, _) in enumerate(train_loader):

                step = e.model.global_step
                if step % e.args.test_log_interval == 0:

                    # Test model
                    with torch.no_grad():
                        self._test(epoch)

                    # Save model checkpoint
                    if step > 0 and step % e.args.checkpoint_interval == 0:
                        print("* saving model checkpoint at step {}".format(step))
                        e.model.checkpoint(self.checkpoint_folder)

                    # Restart progress bar
                    progress = tqdm(total=e.args.test_log_interval, desc='train')

                # Reset gradients
                e.optimizer.zero_grad()

                # Forward pass: get loss and other info
                outputs = e.model_simple_eval(e.model, x)

                # Compute gradients (backward pass)
                outputs['loss'].backward()

                # Add batch metrics to summarizers
                metrics_dict = e.get_metrics_dict(outputs)
                train_summarizers.add(metrics_dict)

                # Update progress bar
                progress.update()

                # Close progress bar if test occurs at next loop iteration
                if (step + 1) % e.args.test_log_interval == 0:
                    if progress is not None:
                        progress.close()


                if (step + 1) % e.args.log_interval == 0:
                    # step+1 because we already did a forward/backward step

                    # Get moving average of training metrics and reset summarizers
                    summaries = train_summarizers.get_all(reset=True)

                    # Print summaries
                    e.print_train_log(step + 1, epoch, summaries)

                    # Add train summaries (smoothed) to history and dump it to
                    # file and to tensorboard if available
                    self.train_history.add(summaries, step)
                    if not e.args.dry_run:
                        with open(self.log_path, 'wb') as fd:
                            pickle.dump(self.train_history.get_dict(), fd)
                        for k, v in summaries.items():
                            self.tb_writer.add_scalar('train_' + k, v, step)

                # Optimization step
                e.optimizer.step()

                # Increment model's global step variable
                e.model.increment_global_step()


    def _test(self, epoch):
        e = self.experiment
        step = e.model.global_step

        # Evaluation mode
        e.model.eval()

        # Get test results
        summaries = e.test_procedure()

        # Additional testing, possibly including saving images
        e.additional_testing(self.img_folder)

        # Print log string with test results (experiment-specific)
        e.print_test_log(summaries, step, epoch)

        # Save summaries to history
        self.test_history.add(summaries, step)

        if not e.args.dry_run:

            # Save summaries to tensorboard
            if self.tb_writer is not None:
                for k, v in summaries.items():
                    self.tb_writer.add_scalar('validation_' + k, v, step)

            # Save history to file
            if self.log_path is not None:
                with open(self.log_path, 'wb') as fd:
                    pickle.dump(self.test_history.get_dict(), fd)

        # Training mode
        e.model.train()
