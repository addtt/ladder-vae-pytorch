from multiobject.pytorch import MultiObjectDataset, MultiObjectDataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CelebA

from lib.datasets import StaticBinaryMnist


multiobject_paths = {
    'multi_mnist_binary': './data/multi_mnist/multi_binary_mnist_012.npz',
    'multi_dsprites_binary_rgb': './data/multi-dsprites-binary-rgb/multi_dsprites_color_012.npz',
}
multiobject_datasets = multiobject_paths.keys()


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

        # Default dataloader class
        dataloader_class = DataLoader

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

        elif args.dataset_name in multiobject_datasets:
            data_path = multiobject_paths[args.dataset_name]
            train_set = MultiObjectDataset(data_path, train=True)
            test_set = MultiObjectDataset(data_path, train=False)

            # Custom data loader class
            dataloader_class = MultiObjectDataLoader

        else:
            raise RuntimeError("Unrecognized data set '{}'".format(args.dataset_name))

        self.train = dataloader_class(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.test = dataloader_class(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]
