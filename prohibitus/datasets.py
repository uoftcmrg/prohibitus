from abc import ABC, abstractmethod
from glob import glob
from itertools import chain, islice
from random import shuffle

from numpy.lib.stride_tricks import sliding_window_view
from torch import long, tensor
from torch.utils.data import IterableDataset

from prohibitus.utilities import load_pro


class Dataset(IterableDataset, ABC):
    def __init__(self, status, configuration):
        if status:
            self.pathname = configuration.training_dataset_pathname
            self.shuffle_count = configuration.training_dataset_shuffle_count
            self.max_dataset_size = configuration.max_training_dataset_size
        else:
            self.pathname = configuration.test_dataset_pathname
            self.shuffle_count = configuration.test_dataset_shuffle_count
            self.max_dataset_size = configuration.max_test_dataset_size

        self.configuration = configuration

    def __iter__(self):
        matches = glob(self.pathname, recursive=True)
        shuffle(matches)

        iterable = chain.from_iterable(map(self._sub_iter, matches))
        count = 0

        while self.max_dataset_size is None or count < self.max_dataset_size:
            batch = list(islice(iterable, self.shuffle_count))

            if not batch:
                return

            shuffle(batch)

            if self.max_dataset_size is not None:
                batch = batch[:self.max_dataset_size - count]

            yield from batch
            count += len(batch)

    def __len__(self):
        return self.max_dataset_size

    @abstractmethod
    def _sub_iter(self, filename):
        ...

    def _sub_iter_aux(self, content):
        if len(content) < self.configuration.chunk_size + 1:
            return

        for chunk in sliding_window_view(
                content,
                self.configuration.chunk_size + 1,
                0,
        ):
            x = tensor(chunk[:-1], dtype=long)
            y = tensor(chunk[1:], dtype=long)

            yield x, y


class ABCDataset(Dataset):
    def _sub_iter(self, filename):
        with open(filename, encoding='utf-8') as file:
            chars = list(map(ord, file.read()))

        for i, char in enumerate(chars):
            if not 0 <= char < self.configuration.token_count:
                chars[i] = 0

        return self._sub_iter_aux(chars)


class MidiDataset(Dataset):
    def _sub_iter(self, filename):
        pro = load_pro(filename, self.configuration)

        return self._sub_iter_aux(pro)
