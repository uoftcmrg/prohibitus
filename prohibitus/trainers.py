from math import cos, inf, pi
from os import makedirs
from os.path import dirname, exists

import numpy as np
from torch import cuda, load, save, set_grad_enabled
from torch.nn import BCELoss, CrossEntropyLoss, DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Trainer:
    criterion = None

    def __init__(self, model, train_dataset, test_dataset, configuration):
        self.raw_model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.configuration = configuration
        self.optimizer = model.create_optimizer()
        self.learning_rate = configuration.learning_rate
        self.token_count = 0
        self.epoch_count = 0
        self.min_test_loss = inf

        if cuda.is_available():
            self.device = cuda.current_device()
            self.model = DataParallel(self.raw_model).to(self.device)
        else:
            self.device = 'cpu'
            self.model = self.raw_model

        if configuration.checkpoint_path is not None \
                and exists(configuration.checkpoint_path):
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = load(self.configuration.checkpoint_path)

        self.raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learning_rate = checkpoint['learning_rate']
        self.token_count = checkpoint['token_count']
        self.epoch_count = checkpoint['epoch_count']
        self.min_test_loss = checkpoint['min_test_loss']

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'token_count': self.token_count,
            'epoch_count': self.epoch_count,
            'min_test_loss': self.min_test_loss,
        }

        dirname_ = dirname(self.configuration.checkpoint_path)

        if not exists(dirname_):
            makedirs(dirname_)

        save(checkpoint, self.configuration.checkpoint_path)

    def train(self):
        if self.configuration.checkpoint_path is not None \
                and not exists(self.configuration.checkpoint_path):
            self.save_checkpoint()

        while self.epoch_count < self.configuration.max_epoch_count:
            self._run_epoch(True)

            if self.test_dataset is None:
                test_loss = None
            else:
                test_loss = self._run_epoch(False)

            self.epoch_count += 1

            if self.configuration.checkpoint_path is not None \
                    and (test_loss is None or test_loss < self.min_test_loss):
                self.min_test_loss = test_loss
                self.save_checkpoint()

    def _run_epoch(self, status):
        self.model.train(status)

        if status:
            label = 'Train'
            dataset = self.train_dataset
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
                logits = self.model(x, False)
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
                f'Epoch {self.epoch_count} | {label} loss {loss.item():.5f}'
                f', Learning rate {self.learning_rate:e}',
            )

        loss = np.mean(losses)

        progress_bar.set_description(
            f'Epoch {self.epoch_count} | {label} loss {loss:.5f}'
            f', Learning rate {self.learning_rate:e}',
        )

        return loss


class ABCTrainer(Trainer):
    criterion = CrossEntropyLoss()


class MidiTrainer(Trainer):
    criterion = BCELoss()
