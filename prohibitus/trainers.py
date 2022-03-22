from math import cos, inf, pi
from os import makedirs
from os.path import dirname, exists

import numpy as np
from torch import cuda, load, save, set_grad_enabled
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model, training_dataset, test_dataset, configuration):
        self.raw_model = model
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.configuration = configuration
        self.criterion = CrossEntropyLoss()
        self.optimizer = model.create_optimizer()
        self.learning_rate = configuration.learning_rate
        self.token_count = 0
        self.epoch_count = 0
        self.min_test_loss = inf
        self.training_losses = []
        self.test_losses = []

        if cuda.is_available():
            self.device = cuda.current_device()
            self.model = DataParallel(self.raw_model).to(self.device)
        else:
            self.device = 'cpu'
            self.model = self.raw_model

        if configuration.checkpoint_pathname is not None \
                and exists(configuration.checkpoint_pathname):
            self.load(True)

    def load(self, status):
        if status:
            pathname = self.configuration.checkpoint_pathname
        else:
            pathname = self.configuration.autosave_pathname

        checkpoint = load(pathname)

        self.raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learning_rate = checkpoint['learning_rate']
        self.token_count = checkpoint['token_count']
        self.epoch_count = checkpoint['epoch_count']
        self.min_test_loss = checkpoint['min_test_loss']
        self.training_losses = checkpoint['training_losses']
        self.test_losses = checkpoint['test_losses']

    def save(self, status):
        checkpoint = {
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'token_count': self.token_count,
            'epoch_count': self.epoch_count,
            'min_test_loss': self.min_test_loss,
            'training_losses': self.training_losses,
            'test_losses': self.test_losses,
        }

        if status:
            pathname = self.configuration.checkpoint_pathname
        else:
            pathname = self.configuration.autosave_pathname

        dirname_ = dirname(pathname)

        if not exists(dirname_):
            makedirs(dirname_)

        save(checkpoint, pathname)

    def train(self):
        if self.configuration.autosave_pathname is not None:
            if exists(self.configuration.autosave_pathname):
                self.load(False)
            else:
                self.save(False)

        if self.configuration.checkpoint_pathname is not None \
                and not exists(self.configuration.checkpoint_pathname):
            self.save(True)

        while self.epoch_count < self.configuration.max_epoch_count:
            training_loss = self._run_epoch(True)

            if self.test_dataset is None:
                test_loss = None
            else:
                test_loss = self._run_epoch(False)

            self.epoch_count += 1
            self.training_losses.append(training_loss)
            self.test_losses.append(test_loss)

            if self.configuration.autosave_pathname is not None:
                self.save(False)

            if self.configuration.checkpoint_pathname is not None \
                    and (test_loss is None or test_loss < self.min_test_loss):
                self.min_test_loss = test_loss
                self.save(True)

    def _run_epoch(self, status):
        self.model.train(status)

        if status:
            label = 'Training'
            dataset = self.training_dataset
        else:
            label = 'Test'
            dataset = self.test_dataset

        loader = DataLoader(
            dataset,
            self.configuration.batch_size,
            pin_memory=True,
        )
        losses = []
        progress_bar = tqdm(loader)

        for x, y in progress_bar:
            x = x.to(self.device)
            y = y.to(self.device)

            with set_grad_enabled(status):
                logits = self.model(x)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                ).mean()

                losses.append(loss.item())

            if status:
                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    self.model.parameters(),
                    self.configuration.grad_norm_clip,
                )
                self.optimizer.step()

                if self.configuration.decay_learning_rate:
                    self.token_count += (y >= 0).sum()

                    w = self.configuration.warmup_token_count
                    f = self.configuration.final_token_count

                    if self.token_count < w:
                        scalar = self.token_count / max(1, w)
                    elif w <= self.token_count < f:
                        progress = (self.token_count - w) / max(1, f - w)
                        scalar = max(0.1, 0.5 * (1 + cos(pi * progress)))
                    else:
                        scalar = 0.1

                    self.learning_rate = \
                        self.configuration.learning_rate * scalar

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            progress_bar.set_description(
                f'Epoch {self.epoch_count} | {label} loss: {losses[-1]:.5f}, '
                f'Learning rate: {self.learning_rate:e}',
            )

        loss = np.mean(losses)

        progress_bar.set_description(
            f'Epoch {self.epoch_count} | {label} loss: {loss:.5f}, '
            f'Learning rate: {self.learning_rate:e}',
        )

        return loss


class ABCTrainer(Trainer):
    ...


class MidiTrainer(Trainer):
    ...
