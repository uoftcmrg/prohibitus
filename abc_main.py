from argparse import ArgumentParser
from random import choices

from torch import cat, long, multinomial, no_grad, tensor, topk
from torch.nn.functional import softmax

from prohibitus import ABCConfiguration, ABCDataset, ABCModel, ABCTrainer


@no_grad()
def infer(model, context, count, configuration, device, *, k=None):
    x = tensor(
        tuple(map(ord, context)),
        dtype=long,
    ).unsqueeze(0).to(device)

    model.eval()

    for _ in range(count):
        input_ = x[:, -configuration.chunk_size:]
        logits = model(input_)
        logits = logits[:, -1, :]
        probabilities = softmax(logits, dim=-1)

        if k is None:
            y = multinomial(probabilities, num_samples=1)
        else:
            values, indices = topk(probabilities[0], k)
            y = tensor([choices(indices, values)]).to(device)

        x = cat((x, y), dim=1)

    completion = ''.join(map(chr, x[0]))

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
    elif args.command == 'infer':
        print(
            infer(
                model,
                'X:40\nL:1/8\nQ:1/4=132\nM:4/4\nI:linebreak $\nK:F\nV:1 treble\nV:2 bass\nL:1/4\nV:1',
                10000,
                configuration,
                trainer.device,
                k=10,
            ),
        )
    else:
        print(f'unknown command: {args.command}')


if __name__ == '__main__':
    main()
