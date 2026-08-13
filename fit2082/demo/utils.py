# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

from typing import Any

import numpy as np
import torch

# == Dataset ===================================================================


class Dataset:
    def __init__(self, path_X, path_Y, batch_size=256, shuffle=True, **kwargs):

        self.path_X = path_X
        self.path_Y = path_Y

        self.batch_size = batch_size

        self._shuffle = shuffle

        self._mmap_X: Any = np.load(path_X, mmap_mode="r")
        self._mmap_Y: Any = np.load(path_Y, mmap_mode="r")

        self._indices = kwargs.get("indices", torch.arange(self._mmap_X.shape[0]))

        self.is_open = True

        # self._reset()

    def __getitem__(self, key):

        return Dataset(
            path_X=self.path_X,
            path_Y=self.path_Y,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            indices=self._indices[key],
        )

    def open(self):

        if not self.is_open:
            self._mmap_X = np.load(self.path_X, mmap_mode="r")
            self._mmap_Y = np.load(self.path_Y, mmap_mode="r")

            self.is_open = True

    def close(self):

        if self.is_open:
            self._mmap_X._mmap.close()
            self._mmap_Y._mmap.close()

            del self._mmap_X
            del self._mmap_Y

            self._mmap_X = None
            self._mmap_Y = None

            self.is_open = False

    @property
    def classes(self):

        self.open()

        return np.unique(self._mmap_Y[self._indices])

    @property
    def shape(self):

        self.open()

        return self._indices.shape[0], *self._mmap_X.shape[1:]

    def _reset(self):

        if self._shuffle:
            _batches = torch.randperm(self._indices.shape[0])
        else:
            _batches = torch.arange(self._indices.shape[0])

        self._batches = _batches.split(self.batch_size)

        self._num_batches = len(self._batches)
        self._batch_index = 0

    def __iter__(self):

        self._reset()

        return self

    def __next__(self):

        self.open()

        if self._batch_index < self._num_batches:
            X = self._mmap_X[self._indices[self._batches[self._batch_index]]]
            Y = self._mmap_Y[self._indices[self._batches[self._batch_index]]]

            if X.ndim < self._mmap_X.ndim:
                X = X.reshape(1, *X.shape)
                Y = np.atleast_1d(Y)

            self._batch_index += 1

            return X, Y

        else:
            raise StopIteration

    @property
    def Y(self):

        self.open()

        return self._mmap_Y[self._indices]
