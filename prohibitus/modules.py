from functools import partial
from math import inf, sqrt
from operator import getitem

from torch import ones, softmax, tril, zeros
from torch.nn import (
    Dropout,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    Module,
    Parameter,
    Sequential,
)
from torch.optim import AdamW


class ProhibitusModule(Module):
    def __init__(self, configuration):
        super().__init__()

        self.configuration = configuration


class CausalSelfAttention(ProhibitusModule):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.key = Linear(
            configuration.embedding_size,
            configuration.embedding_size,
        )
        self.query = Linear(
            configuration.embedding_size,
            configuration.embedding_size,
        )
        self.value = Linear(
            configuration.embedding_size,
            configuration.embedding_size,
        )

        self.attention_dropout = Dropout(
            configuration.attention_drop_percentage,
        )
        self.residual_dropout = Dropout(
            configuration.residual_drop_percentage,
        )

        self.project = Linear(
            configuration.embedding_size,
            configuration.embedding_size,
        )

        self.register_buffer(
            'mask',
            tril(
                ones(configuration.chunk_size, configuration.chunk_size),
            ).view(1, 1, configuration.chunk_size, configuration.chunk_size),
        )

    def forward(self, x):
        batch_size, chunk_size, embedding_size = x.size()
        dimensions = (
            batch_size,
            chunk_size,
            self.configuration.head_count,
            embedding_size // self.configuration.head_count,
        )

        key = self.key(x).view(dimensions).transpose(1, 2)
        query = self.query(x).view(dimensions).transpose(1, 2)
        value = self.value(x).view(dimensions).transpose(1, 2)

        attention = (query @ key.transpose(-2, -1)) / sqrt(embedding_size)
        attention = attention.masked_fill(
            self.mask[:, :, :chunk_size, :chunk_size] == 0,
            -inf,
        )
        attention = softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        y = attention @ value
        y = y.transpose(1, 2).contiguous().view(*x.size())
        y = self.residual_dropout(self.project(y))

        return y


class Block(ProhibitusModule):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.layer_norm_1 = LayerNorm(configuration.embedding_size)
        self.layer_norm_2 = LayerNorm(configuration.embedding_size)
        self.attention = CausalSelfAttention(configuration)
        self.multilayer_perceptron = Sequential(
            Linear(
                configuration.embedding_size,
                configuration.feedforward_size,
            ),
            GELU(),
            Linear(
                configuration.feedforward_size,
                configuration.embedding_size,
            ),
            Dropout(configuration.residual_drop_percentage),
        )

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.multilayer_perceptron(self.layer_norm_2(x))

        return x


class Model(ProhibitusModule):
    @staticmethod
    def _setup(module):
        if isinstance(module, (Linear, Embedding)):
            module.weight.data.normal_(mean=0, std=0.02)

            if isinstance(module, Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1)

    def __init__(self, configuration):
        super().__init__(configuration)

        # Embedder
        self.token_embedder = Embedding(
            configuration.token_count,
            configuration.embedding_size,
        )
        self.positional_embedder = Parameter(
            zeros(1, configuration.chunk_size, configuration.embedding_size),
        )
        self.dropout = Dropout(configuration.embedding_drop_percentage)

        # Transformer
        self.transformer = Sequential(
            *(Block(configuration) for _ in range(configuration.layer_count)),
        )

        # Decoder
        self.layer_norm = LayerNorm(configuration.embedding_size)
        self.decoder = Linear(
            configuration.embedding_size,
            configuration.token_count,
            bias=False,
        )

        self.apply(self._setup)

    def create_optimizer(self):
        decay = set()
        no_decay = {'positional_embedder'}

        whitelist = Linear,
        blacklist = LayerNorm, Embedding

        for module_name, module in self.named_modules():
            for parameter_name, parameter in module.named_parameters():
                if module_name:
                    name = f'{module_name}.{parameter_name}'
                else:
                    name = parameter_name

                if parameter_name.endswith('bias'):
                    no_decay.add(name)
                elif parameter_name.endswith('weight') \
                        and isinstance(module, whitelist):
                    decay.add(name)
                elif parameter_name.endswith('weight') \
                        and isinstance(module, blacklist):
                    no_decay.add(name)

        parameters = dict(self.named_parameters())

        if decay & no_decay:
            raise ValueError('Parameter marked both decay and no decay')
        elif parameters.keys() - (decay | no_decay):
            raise ValueError('Unmarked parameter')

        groups = (
            {
                'params': map(partial(getitem, parameters), sorted(decay)),
                'weight_decay': self.configuration.weight_decay,
            },
            {
                'params': map(partial(getitem, parameters), sorted(no_decay)),
                'weight_decay': 0,
            },
        )
        optimizer = AdamW(
            groups,
            lr=self.configuration.learning_rate,
            betas=self.configuration.betas,
        )

        return optimizer

    def forward(self, x, normalize=True):
        _, chunk_size = x.size()

        # Embedder
        token_embeddings = self.token_embedder(x)
        position_embeddings = self.positional_embedder[:, :chunk_size, :]
        x = self.dropout(token_embeddings + position_embeddings)

        # Transformer
        x = self.transformer(x)

        # Decoder
        x = self.layer_norm(x)
        logits = self.decoder(x)

        return softmax(logits, -1) if normalize else logits


class ABCModel(Model):
    ...


class MidiModel(Model):
    ...
