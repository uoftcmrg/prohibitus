from abc import ABC, abstractmethod
from glob import iglob
from itertools import chain, islice
from random import shuffle

from numpy.lib.stride_tricks import sliding_window_view
from torch import long, tensor
from torch.utils.data.dataset import IterableDataset

from prohibitus.utilities import load_piano_roll


class Dataset(IterableDataset, ABC):
    def __init__(self, status, configuration):
        if status:
            self.pathname = configuration.train_pathname
        else:
            self.pathname = configuration.test_pathname

        self.configuration = configuration

    def __iter__(self):
        matches = iglob(self.pathname, recursive=True)

        iterable = chain.from_iterable(map(self._sub_iter, matches))

        batch = None

        while batch or batch is None:
            batch = list(islice(iterable, self.configuration.shuffle_count))
            shuffle(batch)

            yield from batch

    @abstractmethod
    def _sub_iter(self, filename):
        ...


class ABCDataset(Dataset):
    def _sub_iter(self, filename):
        with open(filename, encoding='utf-8') as file:
            chars = list(map(ord, file.read()))

        for i, char in enumerate(chars):
            if not 0 <= char < self.configuration.token_count:
                chars[i] = 0

        for i in range(len(chars) - self.configuration.chunk_size):
            chunk = chars[i:i + self.configuration.chunk_size + 1]

            x = tensor(chunk[:-1], dtype=long)
            y = tensor(chunk[1:], dtype=long)

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
            x = tensor(chunk[:-1])
            y = tensor(chunk[1:])

            yield x, y
