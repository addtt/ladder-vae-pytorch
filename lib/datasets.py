import os
from urllib import request

import numpy as np
import torch
from torch.utils.data import TensorDataset


class StaticBinaryMnist(TensorDataset):

    def __init__(self, folder, train, download=False, shuffle_init=False):
        self.download = download
        if train:
            sets = [
                self._get_binarized_mnist(folder, shuffle_init, split='train'),
                self._get_binarized_mnist(folder, shuffle_init, split='valid')
            ]
            x = np.concatenate(sets, axis=0)
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

        fname = 'binarized_mnist_{}.npz'.format(split)
        path = os.path.join(folder, fname)

        if not os.path.exists(path):
            print("Dataset file '{}' not found".format(path))
            if not self.download:
                msg = "Dataset not found, use download=True to download it"
                raise RuntimeError(msg)

            print("Downloading whole dataset...")

            os.makedirs(folder, exist_ok=True)

            for subdataset in subdatasets:
                fname_mat = 'binarized_mnist_{}.amat'.format(subdataset)
                url = ('http://www.cs.toronto.edu/~larocheh/public/datasets/'
                       'binarized_mnist/{}'.format(fname_mat))
                path_mat = os.path.join(folder, fname_mat)
                request.urlretrieve(url, path_mat)

                with open(path_mat) as f:
                    lines = f.readlines()

                os.remove(path_mat)
                lines = np.array(
                    [[int(i) for i in line.split()] for line in lines])
                data[subdataset] = lines.astype('float32').reshape(
                    (-1, 1, 28, 28))
                np.savez_compressed(path_mat.split(".amat")[0],
                                    data=data[subdataset])

        else:
            data[split] = np.load(path)['data']

        if shuffle_init:
            np.random.shuffle(data[split])

        return data[split]


def _pad_tensor(x, size, value=None):
    assert isinstance(x, torch.Tensor)
    input_size = len(x)
    if value is None:
        value = float('nan')

    # Copy input tensor into a tensor filled with specified value
    # Convert everything to float, not ideal but it's robust
    out = torch.zeros(*size, dtype=torch.float)
    out.fill_(value)
    if input_size > 0:  # only if at least one element in the sequence
        out[:input_size] = x.float()
    return out
