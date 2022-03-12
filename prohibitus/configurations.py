from abc import ABC


class Configuration(ABC):
    # Data settings
    train_pathname = None
    test_pathname = None
    shuffle_count = None

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
    autosave_interval = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unknown attribute: {key}')

        assert self.embedding_size % self.head_count == 0


class ABCConfiguration(Configuration):
    # Data settings
    train_pathname = './resources/abc/*.abc'
    test_pathname = None
    shuffle_count = 1000000

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
    checkpoint_path = './checkpoints/abc/checkpoint.pt'
    autosave_path = './checkpoints/abc/autosave.pt'
    autosave_interval = 1000


class MidiConfiguration(Configuration):
    # Data settings
    train_pathname = './resources/midi/train/**/*.mid'
    test_pathname = './resources/midi/test/**/*.mid'
    shuffle_count = 1e7
    threshold = 0.01
    framerate = 128

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
    checkpoint_path = './checkpoints/midi/checkpoint.pt'
    autosave_path = './checkpoints/midi/autosave.pt'
    autosave_interval = 1000
