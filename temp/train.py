"""Copyright (c) 2022 - Juho Kim"""

import logging

import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from prohibitus import ProhibitusConfiguration, ProhibitusModel, \
    ProhibitusTrainer

# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_chunk_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


# make deterministic
set_seed(42)


class CharDataset(Dataset):
    def __init__(self, data, chunk_size):
        chars = sorted(list(set(data)))
        data_size, token_dim = len(data), len(chars)
        print('data has %d characters, %d unique.' % (
            data_size, token_dim))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.chunk_size = chunk_size
        self.token_dim = token_dim
        self.data = data

    def __len__(self):
        return len(self.data) - self.chunk_size

    def __getitem__(self, idx):
        # grab a chunk of (chunk_size + 1) characters from the data
        chunk = self.data[idx:idx + self.chunk_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


chunk_size = 128  # spatial extent of the model for its context
train_dataset = CharDataset(open('input.txt').read(), chunk_size)

configuration = ProhibitusConfiguration(
    token_count=train_dataset.token_dim,
    chunk_size=train_dataset.chunk_size,
    layer_count=4,
    head_count=4,
    embedding_count=128,
    feedforward_count=512,
    max_epochs=2, batch_size=512, learning_rate=6e-4,
    lr_decay=True, warmup_tokens=512 * 20,
    final_tokens=2 * len(train_dataset) * chunk_size,
    num_workers=0,
)
model = ProhibitusModel(configuration)

# initialize a trainer instance and kick off training
trainer = ProhibitusTrainer(model, train_dataset, None, configuration)
trainer.train()

torch.save(model.state_dict(), 'model.pt')

context = """Claudio:
Who goes there?
"""
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
    None, ...].to(trainer.device)
y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
