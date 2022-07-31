# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data.dataloader import default_collate
import numpy as np

from . import BaseWrapperDataset, plasma_utils


class ResamplingDataset(BaseWrapperDataset):
    """Randomly samples from a given dataset at each epoch.

    Sampling is done with or without replacement, depending on the "replace"
    parameter.

    Optionally, the epoch size can be rescaled. This is potentially desirable
    to increase per-epoch coverage of the base dataset (since sampling with
    replacement means that many items in the dataset will be left out). In the
    case of sampling without replacement, size_ratio should be strictly less
    than 1.

    Args:
        dataset (~torch.utils.data.Dataset): dataset on which to sample.
        weights (List[float]): list of probability weights
            (default: None, which corresponds to uniform sampling).
        replace (bool): sampling mode; True for "with replacement", or False
            for "without replacement" (default: True)
        size_ratio (float): the ratio to subsample to; must be positive
            (default: 1.0).
        batch_by_size (bool): whether or not to batch by sequence length
            (default: True).
        seed (int): RNG seed to use (default: 0).
        epoch (int): starting epoch number (default: 1).
    """

    def __init__(
        self,
        dataset,
        weights=None,
        replace=True,
        size_ratio=1.0,
        batch_by_size=True,
        seed=0,
        epoch=1,
    ):
        super().__init__(dataset)
        self.dataset = dataset

        if weights is None:
            self.weights = None

        else:
            assert len(weights) == len(dataset)
            weights_arr = np.array(weights, dtype=np.float64)
            weights_arr /= weights_arr.sum()
            self.weights = plasma_utils.PlasmaArray(weights_arr)

        self.replace = replace

        assert size_ratio > 0.0
        if not self.replace:
            assert size_ratio <= 1.0
        self.size_ratio = float(size_ratio)
        self.orig_size = len(dataset)
        self.actual_size = np.ceil(len(dataset) * self.size_ratio).astype(int)

        self.batch_by_size = batch_by_size
        self.seed = seed

        self._cur_epoch = None
        self._cur_indices = None

        self.set_epoch(epoch)

    def __getitem__(self, index):
        return self.dataset[self._cur_indices.array[index]]

    def __len__(self):
        return self.actual_size

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        if isinstance(self.dataset.sizes, list):
            return [s[self._cur_indices.array] for s in self.dataset.sizes]
        return self.dataset.sizes[self._cur_indices.array]

    def num_tokens(self, index):
        return self.dataset.num_tokens(self._cur_indices.array[index])

    def size(self, index):
        return self.dataset.size(self._cur_indices.array[index])

    def ordered_indices(self):
        if self.batch_by_size:
            order = [
                np.arange(len(self)),
                self.sizes,
            ]  # No need to handle `self.shuffle == True`
            return np.lexsort(order)
        else:
            return np.arange(len(self))

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.dataset.prefetch(self._cur_indices.array[indices])

    def reset_size_ratio(self, size_ratio):
        self.size_ratio = float(size_ratio)
        if self.size_ratio > 1.0:
            self.replace = True
        self.actual_size = np.ceil(self.orig_size * self.size_ratio).astype(int)

    def set_epoch(self, epoch, **kwargs):
        super().set_epoch(epoch)
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

        if epoch == self._cur_epoch:
            return

        self._cur_epoch = epoch

        if kwargs.get('size_ratio') is not None:
            self.reset_size_ratio(kwargs['size_ratio'])
            
        # Generate a weighted sample of indices as a function of the
        # random seed and the current epoch.

        rng = np.random.RandomState(
            [
                42,  # magic number
                self.seed % (2 ** 32),  # global seed
                self._cur_epoch,  # epoch index
            ]
        )
        self._cur_indices = plasma_utils.PlasmaArray(
            rng.choice(
                len(self.dataset),
                self.actual_size,
                replace=self.replace,
                p=(None if self.weights is None else self.weights.array),
            )
        )
