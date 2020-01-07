from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CelebA

from datasets import MultiObjectDataset, MultiObjectDataLoader, StaticBinaryMnist


class DatasetLoader:
    """
    Wrapper for DataLoaders. Data attributes:
    - train: DataLoader object for training set
    - test: DataLoader object for test set
    - data_shape: shape of each data point (channels, height, width)
    - img_size: spatial dimensions of each data point (height, width)
    - color_ch: number of color channels
    """

    def __init__(self, args, cuda):

        kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}

        # Init dataloaders to None
        self.train = self.test = None

        if args.dataset_name == 'static_mnist':
            data_folder = './data/static_bin_mnist/'
            train_set = StaticBinaryMnist(data_folder, train=True,
                                          download=True, shuffle_init=True)
            test_set = StaticBinaryMnist(data_folder, train=False,
                                         download=True, shuffle_init=True)

        elif args.dataset_name == 'cifar10':
            # Discrete values 0, 1/255, ..., 254/255, 1
            transform = transforms.Compose([
                # Move values to the center of 256 bins
                # transforms.Lambda(lambda x: Image.eval(
                #     x, lambda y: y * (255/256) + 1/512)),
                transforms.ToTensor(),
            ])
            data_folder = './data/cifar10/'
            train_set = CIFAR10(data_folder, train=True,
                                download=True, transform=transform)
            test_set = CIFAR10(data_folder, train=False,
                               download=True, transform=transform)

        elif args.dataset_name == 'svhn':
            transform = transforms.ToTensor()
            data_folder = './data/svhn/'
            train_set = SVHN(data_folder, split='train',
                             download=True, transform=transform)
            test_set = SVHN(data_folder, split='test',
                            download=True, transform=transform)

        elif args.dataset_name == 'celeba':
            transform = transforms.Compose([
                transforms.CenterCrop(148),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
            data_folder = '/scratch/adit/data/celeba/'
            train_set = CelebA(data_folder, split='train',
                               download=True, transform=transform)
            test_set = CelebA(data_folder, split='valid',
                              download=True, transform=transform)

        elif args.dataset_name == 'multi_dsprites_binary_rgb':
            data_path = './data/multi-dsprites-binary-rgb/multi_dsprites_color_012.npz'
            train_set = MultiObjectDataset(data_path, train=True)
            test_set = MultiObjectDataset(data_path, train=False)

            # Custom data loaders
            self.train = MultiObjectDataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                **kwargs
            )
            self.test = MultiObjectDataLoader(
                test_set,
                batch_size=args.test_batch_size,
                shuffle=False,
                **kwargs
            )

        else:
            raise RuntimeError("Unrecognized data set '{}'".format(args.dataset_name))

        # Default training set loader if it hasn't been defined yet
        if self.train is None:
            self.train = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                **kwargs
            )

        # Default test set loader if it hasn't been defined yet
        if self.test is None:
            self.test = DataLoader(
                test_set,
                batch_size=args.test_batch_size,
                shuffle=False,
                **kwargs
            )

        self.data_shape = self.train.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]
