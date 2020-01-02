class BaseExperimentConfig:
    """
    Experiment configuration.

    Data attributes:

    - 'args': argparse.Namespace containing all config parameters. When
      initializing the ExperimentConfig, if 'args' is not given, all config
      parameters are set based on experiment defaults and user input, using
      argparse.

    - 'run_description': string description of the run that includes a timestamp
      and can be used e.g. as folder name for logging.

    - 'model': the model.

    - 'device': torch.device that is being used

    - 'dataloaders': DataLoaders, with attributes 'train' and 'test'

    - 'optimizer': the optimizer

    """
    def __init__(self, args=None):
        self.device = None
        self.dataloaders = None
        self.model = None
        self.optimizer = None
        self.args = args
        self.max_epochs = 100000   # no limit
        if args is None:
            self.args = self._parse_args()
        self.run_description = self._make_run_description(self.args)


    @staticmethod
    def _parse_args():
        """
        Parse command-line arguments defining experiment settings.

        :return: args: argparse.Namespace with experiment settings
        """
        raise NotImplementedError


    @staticmethod
    def _make_run_description(args):
        """
        Create a string description of the run. It is used in the names of the
        logging folders.

        :param args: experiment config
        :return: the run description
        """
        raise NotImplementedError


    def setup(self, device, create_optimizer=True):
        """
        Set up experiment. Define data loaders, model, and optimizer.

        :param device:
        :param create_optimizer:
        """
        raise NotImplementedError


    def basic_model_eval(self, model, x):
        """
        Simple single-pass model evaluation. It consists of a forward pass
        and computation of all necessary losses and metrics.

        :param model:
        :param x:
        :return:
        """
        raise NotImplementedError


    @staticmethod
    def print_train_log(step, epoch, summaries):
        raise NotImplementedError


    @staticmethod
    def print_test_log(summaries, step=None, epoch=None):
        raise NotImplementedError


    @staticmethod
    def get_metrics_dict(results):
        raise NotImplementedError


    def test_procedure(self, **kwargs):
        """
        Execute test procedure for the experiment. This typically includes
        collecting metrics on the test set using model_simple_eval().
        For example in variational inference we might be interested in
        repeating this many times to derive the importance-weighted ELBO.
        """
        raise NotImplementedError


    def additional_testing(self, img_folder):
        """
        Perform additional testing, including possibly generating images.

        :param img_folder: folder to store images
        """
        raise NotImplementedError
