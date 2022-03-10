class ProhibitusConfiguration:
    # Model settings
    attention_drop_percentage = 0.1
    residual_drop_percentage = 0.1
    embedding_drop_percentage = 0.1
    token_count = 128
    chunk_size = 512
    embedding_size = 768
    feedforward_size = 3072
    head_count = 12
    layer_count = 12

    # Trainer settings
    learning_rate = 3e-4
    betas = 0.9, 0.95
    weight_decay = 0.1
    max_epoch_count = 10
    batch_size = 64
    grad_norm_clip = 1.0
    decay_learning_rate = False
    warmup_token_count = 375e6
    final_token_count = 260e9
    checkpoint_path = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unknown attribute: {key}')

        assert self.embedding_size % self.head_count == 0


class ABCConfiguration(ProhibitusConfiguration):
    ...


class MidiConfiguration(ProhibitusConfiguration):
    # Data settings
    threshold = 0.01
    framerate = 128
