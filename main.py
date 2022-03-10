"""Copyright (c) 2022 - Juho Kim"""

import torch

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample

from glob import glob

contents = []

for filename in glob('resources/abc/*.abc'):
    with open(filename) as file:
        contents.append(file.read())

train_dataset = CharDataset(''.join(contents),
                            block_size)  # one line of poem is roughly 50 characters

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=4, n_head=4, n_embd=128)
model = GPT(mconf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=2, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * block_size,
                      num_workers=0)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()

torch.save(model.state_dict(), 'model.pt')

context = "X:1"
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
    None, ...].to(trainer.device)
y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)


def train():
    ...


def infer():
    ...


def main():
    ...


if __name__ == '__main__':
    main()
