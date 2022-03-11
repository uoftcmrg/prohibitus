class Configuration:
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

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unknown attribute: {key}')

        assert self.embedding_size % self.head_count == 0


class ABCConfiguration(Configuration):
    # Model settings
    attention_drop_percentage = 0.1
    residual_drop_percentage = 0.1
    embedding_drop_percentage = 0.1
    token_count = 128
    chunk_size = 512
    embedding_size = 256
    feedforward_size = 1024
    head_count = 8
    layer_count = 6

    # Trainer settings
    learning_rate = 5e-4
    betas = 0.9, 0.95
    weight_decay = 0.1
    max_epoch_count = 20
    batch_size = 512
    grad_norm_clip = 1.0
    decay_learning_rate = True
    warmup_token_count = 5e6
    final_token_count = 1e7
    checkpoint_path = './models/abc_checkpoint.pt'


class MidiConfiguration(Configuration):
    # Trainer settings
    checkpoint_path = './models/midi_checkpoint.pt'

    # Data settings
    threshold = 0.01
    framerate = 128