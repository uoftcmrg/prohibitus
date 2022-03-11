from argparse import ArgumentParser

from torch import cat, long, multinomial, no_grad, tensor
from torch.nn.functional import softmax

from prohibitus import ABCConfiguration, ABCDataset, ABCModel, ABCTrainer


@no_grad()
def infer(model, context, count, configuration):
    x = tensor(
        tuple(map(ord, context)),
        dtype=long,
    ).unsqueeze(0).to(model.device)

    model.eval()

    for k in range(count):
        input_ = x[:, -configuration.chunk_size:]
        logits = model(input_)
        logits = logits[:, -1, :]
        probabilities = softmax(logits, dim=-1)
        y = multinomial(probabilities, num_samples=1)
        x = cat((x, y), dim=1)

    completion = ''.join(map(chr, x))

    return completion


def main():
    parser = ArgumentParser(
        description='Train or infer music generation in abc format',
    )
    parser.add_argument('command', metavar='<command>')

    args = parser.parse_args()

    configuration = ABCConfiguration()
    model = ABCModel(configuration)
    train_dataset = ABCDataset(True, configuration)
    test_dataset = None
    trainer = ABCTrainer(model, train_dataset, test_dataset, configuration)

    if args.command == 'train':
        trainer.train()
    else:
        print(infer(model, 'X', 10000, configuration))


if __name__ == '__main__':
    main()
