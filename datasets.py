import os
import tarfile
from urllib import request

import numpy as np
import torch
from torch.utils.data import TensorDataset

# TODO move to utils
def untargz(path):
    folder = os.path.dirname(path)
    tar = tarfile.open(path, "r:gz")
    tar.extractall(folder)
    tar.close()


class MultiDSpritesBinaryRGB(TensorDataset):

    def __init__(self, data_path, train, shuffle_init=False, split=0.9):
        train_x, test_x = self.get_data_from_file(data_path, split)
        x = train_x if train else test_x
        x = np.transpose(x, [0, 3, 1, 2])
        assert x.shape[1] == 3
        if shuffle_init:
            np.random.shuffle(x)
        labels = torch.zeros(len(x),).fill_(float('nan'))
        super().__init__(torch.from_numpy(x), labels)

    @staticmethod
    def get_data_from_file(path, split):
        # The string path is like folder/name (no trailing /)
        # The archive is folder/name.tar.gz
        # The data is folder/name/name.npz
        name = os.path.basename(path)
        npz_path = os.path.join(path, name + '.npz')
        if not os.path.isfile(npz_path):  # no numpy file
            print("Dataset not found: decompressing targz")
            untargz(path + '.tar.gz')
        data = np.load(npz_path, allow_pickle=True)
        x = np.array(data['x'], dtype=np.float32) / 255
        split = int(split * len(x))
        train = x[:split]
        test = x[split:]
        return train, test


class StaticBinaryMnist(TensorDataset):

    def __init__(self, folder, train, download=False, shuffle_init=False):
        self.download = download
        if train:
            x = np.concatenate([
                self._get_binarized_mnist(folder, shuffle_init, split='train'),
                self._get_binarized_mnist(folder, shuffle_init, split='valid')
            ], axis=0)
        else:
            x = self._get_binarized_mnist(folder, shuffle_init, split='test')
        labels = torch.zeros(len(x),).fill_(float('nan'))
        super().__init__(torch.from_numpy(x), labels)


    def _get_binarized_mnist(self, folder, shuffle_init, split=None):
        """
        Get statically binarized MNIST. Code partially taken from
        https://github.com/altosaar/proximity_vi/blob/master/get_binary_mnist.py
        """

        subdatasets = ['train', 'valid', 'test']
        if split not in subdatasets:
            raise ValueError("Valid splits: {}".format(subdatasets))
        data = {}

        if not os.path.exists(folder):
            if not self.download:
                msg = "Dataset not found, use download=True to download it"
                raise RuntimeError(msg)

            os.makedirs(folder)

            for subdataset in subdatasets:
                filename = 'binarized_mnist_{}.amat'.format(subdataset)
                url = ('http://www.cs.toronto.edu/~larocheh/public/datasets/'
                       'binarized_mnist/binarized_mnist_{}.amat'.format(subdataset))
                local_filename = os.path.join(folder, filename)
                request.urlretrieve(url, local_filename)

                with open(os.path.join(folder, filename)) as f:
                    lines = f.readlines()

                os.remove(local_filename)
                lines = np.array([[int(i) for i in line.split()] for line in lines])
                data[subdataset] = lines.astype('float32').reshape((-1, 1, 28, 28))
                np.savez_compressed(local_filename.split(".amat")[0], data=data[subdataset])

        else:
            filename = 'binarized_mnist_{}.npz'.format(split)
            local_filename = os.path.join(folder, filename)
            data[split] = np.load(local_filename)['data']

        if shuffle_init:
            np.random.shuffle(data[split])

        return data[split]
