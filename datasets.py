import os
from urllib import request

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data._utils.collate import default_collate


class MultiObjectDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        assert 'collate_fn' not in kwargs
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):

        # The input is a batch of (image, label_dict)
        _, item_labels = batch[0]
        keys = item_labels.keys()

        # Max label length in this batch
        # max_len[k] is the maximum length (in batch) of the label with name k
        # If at the end max_len[k] is -1, labels k are (probably all) scalars
        max_len = {k: -1 for k in keys}

        # If a label has more than 1 dimension, the padded tensor cannot simply
        # have size (batch, max_len). Whenever the length is >0 (i.e. the sequence
        # is not empty, store trailing dimensions. At the end if 1) all sequences
        # (in the batch, and for this label) are empty, or 2) this label is not
        # a sequence (scalar), then the trailing dims are None.
        trailing_dims = {k: None for k in keys}

        # Make first pass to get shape info for padding
        for _, labels in batch:
            for k in keys:
                try:
                    max_len[k] = max(max_len[k], len(labels[k]))
                    if len(labels[k]) > 0:
                        trailing_dims[k] = labels[k].size()[1:]
                except TypeError:   # scalar
                    pass

        # For each item in the batch, take each key and pad the corresponding
        # value (label) so we can call the default collate function
        for i in range(len(batch)):
            for k in keys:
                if trailing_dims[k] is None:
                    continue
                size = [max_len[k]] + list(trailing_dims[k])
                batch[i][1][k] = _pad_tensor(batch[i][1][k], size)

        return default_collate(batch)


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
                lines = np.array([[int(i) for i in line.split()] for line in lines])
                data[subdataset] = lines.astype('float32').reshape((-1, 1, 28, 28))
                np.savez_compressed(path_mat.split(".amat")[0], data=data[subdataset])

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
