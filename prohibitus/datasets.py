from abc import ABC, abstractmethod
from glob import iglob
from itertools import chain

import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data.dataset import IterableDataset

from prohibitus.utilities import load_piano_roll


class Dataset(IterableDataset, ABC):
    def __init__(self, pathname, configuration):
        self.pathname = pathname
        self.configuration = configuration

    def __iter__(self):
        matches = iglob(self.pathname, recursive=True)

        yield from chain.from_iterable(map(self._sub_iter, matches))

    @abstractmethod
    def _sub_iter(self, filename):
        ...


class ABCDataset(Dataset):
    def _sub_iter(self, filename):
        with open(filename) as file:
            content = file.read()

        for i in range(len(content) - self.block_size):
            chunk = self.data[i:i + self.block_size + 1]
            chars = list(map(ord, chunk))

            x = torch.tensor(chars[:-1], dtype=torch.long)
            y = torch.tensor(chars[1:], dtype=torch.long)

            yield x, y


class MidiDataset(Dataset):
    def _sub_iter(self, filename):
        piano_roll = load_piano_roll(filename, self.configuration)

        if piano_roll.shape[0] < self.configuration.chunk_size + 1:
            return

        for chunk in sliding_window_view(
                piano_roll,
                self.configuration.chunk_dim + 1,
                0,
        ):
            x = torch.tensor(chunk[:-1])
            y = torch.tensor(chunk[1:])

            yield x, y
