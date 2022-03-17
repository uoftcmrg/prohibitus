from abc import ABC
from math import inf


class Configuration(ABC):
    # Data settings
    train_dataset_pathname = None
    test_dataset_pathname = None
    dataset_shuffle_count = None
    max_train_dataset_size = None
    max_test_dataset_size = None

    # Model settings
    attention_drop_percentage = None
    residual_drop_percentage = None
    embedding_drop_percentage = None
    token_count = None
    chunk_size = None
    embedding_size = None
    feedforward_size = None
    head_count = None
    layer_count = None

    # Trainer settings
    learning_rate = None
    betas = None
    weight_decay = None
    max_epoch_count = None
    batch_size = None
    grad_norm_clip = None
    decay_learning_rate = None
    warmup_token_count = None
    final_token_count = None
    checkpoint_path = None
    autosave_path = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unknown attribute: {key}')

        assert self.embedding_size % self.head_count == 0


class ABCConfiguration(Configuration):
    # Data settings
    train_dataset_pathname = './resources/abc/*.abc'
    test_dataset_pathname = None
    dataset_shuffle_count = 1000000
    max_train_dataset_size = inf
    max_test_dataset_size = inf

    # Model settings
    attention_drop_percentage = 0.1
    residual_drop_percentage = 0.1
    embedding_drop_percentage = 0.1
    token_count = 128
    chunk_size = 128
    embedding_size = 512
    feedforward_size = 1024
    head_count = 8
    layer_count = 8

    # Trainer settings
    learning_rate = 6e-4
    betas = 0.9, 0.95
    weight_decay = 0.1
    max_epoch_count = 20
    batch_size = 256
    grad_norm_clip = 1.0
    decay_learning_rate = True
    warmup_token_count = 1e5
    final_token_count = 1e9
    checkpoint_path = './saves/abc/checkpoint.pt'
    autosave_path = './saves/abc/autosave.pt'


class MidiConfiguration(Configuration):
    # Data settings
    train_dataset_pathname = './resources/midi/train/**/*.mid'
    test_dataset_pathname = './resources/midi/test/**/*.mid'
    dataset_shuffle_count = 2000000
    max_train_dataset_size = 400000
    max_test_dataset_size = 100000

    # Model settings
    attention_drop_percentage = 0.1
    residual_drop_percentage = 0.1
    embedding_drop_percentage = 0.1
    token_count = 12
    chunk_size = 128
    embedding_size = 512
    feedforward_size = 1024
    head_count = 8
    layer_count = 8

    # Trainer settings
    learning_rate = 6e-4
    betas = 0.9, 0.95
    weight_decay = 0.1
    max_epoch_count = 2
    batch_size = 256
    grad_norm_clip = 1.0
    decay_learning_rate = True
    warmup_token_count = 1e6
    final_token_count = 1e12
    checkpoint_path = './saves/midi/checkpoint.pt'
    autosave_path = './saves/midi/autosave.pt'
