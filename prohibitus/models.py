import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, configuration):
        super().__init__()
        assert configuration.embedding_count % configuration.head_count == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(configuration.embedding_count, configuration.embedding_count)
        self.query = nn.Linear(configuration.embedding_count, configuration.embedding_count)
        self.value = nn.Linear(configuration.embedding_count, configuration.embedding_count)
        # regularization
        self.attention_drop = nn.Dropout(configuration.attention_drop_percentage)
        self.resid_drop = nn.Dropout(configuration.residual_drop_percentage)
        # output projection
        self.proj = nn.Linear(configuration.embedding_count, configuration.embedding_count)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(configuration.chunk_dim, configuration.chunk_dim))
                             .view(1, 1, configuration.chunk_dim, configuration.chunk_dim))
        self.head_count = configuration.head_count

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.head_count, C // self.head_count).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.head_count, C // self.head_count).transpose(
            1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.head_count, C // self.head_count).transpose(
            1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attention_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T,
                                                C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, configuration):
        super().__init__()
        self.ln1 = nn.LayerNorm(configuration.embedding_count)
        self.ln2 = nn.LayerNorm(configuration.embedding_count)
        self.attn = CausalSelfAttention(configuration)
        self.mlp = nn.Sequential(
            nn.Linear(configuration.embedding_count, 4 * configuration.embedding_count),
            nn.GELU(),
            nn.Linear(4 * configuration.embedding_count, configuration.embedding_count),
            nn.Dropout(configuration.residual_drop_percentage),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ProhibitusModel(nn.Module):
    def __init__(self, configuration):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(configuration.token_dim, configuration.embedding_count)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, configuration.chunk_dim, configuration.embedding_count))
        self.drop = nn.Dropout(configuration.embedding_drop_percentage)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(configuration) for _ in range(configuration.layer_count)])
        # decoder head
        self.ln_f = nn.LayerNorm(configuration.embedding_count)
        self.head = nn.Linear(configuration.embedding_count, configuration.token_dim, bias=False)

        self.chunk_dim = configuration.chunk_dim
        self.apply(self._init_weights)

        logger.info("number of parameters: %e",
                    sum(p.numel() for p in self.parameters()))

    def get_chunk_dim(self):
        return self.chunk_dim

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configurationure_optimizers(self, train_configuration):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m,
                                                          whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m,
                                                          blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (
        str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (
                                                    str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": train_configuration.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=train_configuration.learning_rate,
                                      betas=train_configuration.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.chunk_dim, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(
            idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t,
                              :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))

        return logits, loss
