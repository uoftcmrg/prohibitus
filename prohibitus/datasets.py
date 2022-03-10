from glob import iglob

import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data.dataset import IterableDataset

from prohibitus.utilities import load_piano_roll


class GlobDataset(IterableDataset):
    def __init__(self, pathname, configuration):
        self.pathname = pathname
        self.configuration = configuration


class ABCDataset(IterableDataset):
    def __init__(self, pathname, configuration):
        self.pathname = pathname
        self.configuration = configuration

    def __iter__(self):
        filenames = iglob(self.pathname, recursive=True)

        for filename in filenames:
            piano_roll = load_piano_roll(filename, self.configuration)

            if piano_roll.shape[0] < self.configuration.chunk_size + 1:
                continue

            for chunk in sliding_window_view(
                    piano_roll,
                    self.configuration.chunk_dim + 1,
                    0,
            ):
                x = torch.tensor(chunk[:-1])
                y = torch.tensor(chunk[1:])

                yield x, y


class MidiDataset(IterableDataset):
    def __init__(self, pathname, configuration):
        self.pathname = pathname
        self.configuration = configuration

    def __iter__(self):
        filenames = iglob(self.pathname, recursive=True)

        for filename in filenames:
            piano_roll = load_piano_roll(filename, self.configuration)

            if piano_roll.shape[0] < self.configuration.chunk_size + 1:
                continue

            for chunk in sliding_window_view(
                    piano_roll,
                    self.configuration.chunk_dim + 1,
                    0,
            ):
                x = torch.tensor(chunk[:-1])
                y = torch.tensor(chunk[1:])

                yield x, y
