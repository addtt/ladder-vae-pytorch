import os
from urllib import request

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class MultiObjectDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        assert 'collate_fn' not in kwargs
        kwargs['collate_fn'] = self.custom_collate
        super().__init__(*args, **kwargs)

    @staticmethod
    def custom_collate(batch):
        # Input is a batch of (image, label_dict)
        _, item_labels = batch[0]
        keys = item_labels.keys()

        # Simply stack images into a tensor
        imgs = [item[0] for item in batch]
        imgs = torch.stack(imgs, dim=0)

        # All labels will be lists, because they might have variable length
        # across the batch.
        # Item is batch[i], second element is label dict, pick attribute k.
        labels = {
            k: [item[1][k] for item in batch]
            for k in keys}

        return imgs, labels


class MultiObjectDataset(Dataset):

    def __init__(self, data_path, train, split=0.9):
        super().__init__()

        # Load data
        data = np.load(data_path, allow_pickle=True)

        # Rescale images and permute dimensions
        x = np.array(data['x'], dtype=np.float32) / 255
        x = np.transpose(x, [0, 3, 1, 2])  # batch, channels, h, w

        # Get labels
        labels = data['labels'].item()

        # Split train and test
        split = int(split * len(x))
        if train:
            indices = range(split)
        else:
            indices = range(split, len(x))

        # From numpy/ndarray to torch tensors (labels are lists of tensors as
        # they might have different sizes)
        self.x = torch.from_numpy(x[indices])
        self.labels = self._labels_to_tensorlist(labels, indices)

    @staticmethod
    def _labels_to_tensorlist(labels, indices):
        out = {k: [] for k in labels.keys()}
        for i in indices:
            for k in labels.keys():
                t = labels[k][i]
                t = torch.as_tensor(t)
                out[k].append(t)
        return out

    def __getitem__(self, index):
        x = self.x[index]
        labels = {k: self.labels[k][index] for k in self.labels.keys()}
        return x, labels

    def __len__(self):
        return self.x.size(0)


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
