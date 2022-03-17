from argparse import ArgumentParser
from functools import partial
from operator import getitem

from torch import cat, long, multinomial, no_grad, tensor

from prohibitus import (
    MidiConfiguration,
    MidiDataset,
    MidiModel,
    MidiTrainer,
    load_pro,
    save_pro,
)


@no_grad()
def infer(model, context, count, device, configuration):
    x = tensor(
        tuple(map(partial(getitem, MidiDataset.indices), context)),
        dtype=long,
    ).unsqueeze(0).to(device)

    model.eval()

    for _ in range(count):
        input_ = x[:, -configuration.chunk_size:]
        probabilities = model(input_)[:, -1, :]
        y = multinomial(probabilities, num_samples=1)
        x = cat((x, y), dim=1)

    lookup = dict(map(reversed, MidiDataset.indices.items()))

    completion = ''.join(map(partial(getitem, lookup), map(int, x[0])))

    return completion


def main():
    parser = ArgumentParser(
        description='Train or infer music generation in midi format',
    )
    parser.add_argument('command', metavar='<command>')
    parser.add_argument('--filename', metavar='<filename to continue>')
    parser.add_argument(
        '--count',
        type=int,
        metavar='<number of generated characters>',
    )

    args = parser.parse_args()

    configuration = MidiConfiguration()
    model = MidiModel(configuration)
    train_dataset = MidiDataset(True, configuration)
    test_dataset = MidiDataset(False, configuration)
    trainer = MidiTrainer(model, train_dataset, test_dataset, configuration)

    if args.command == 'train':
        trainer.train()
    elif args.command == 'infer':
        pro = load_pro(args.filename)

        pro = infer(
            model,
            pro,
            args.count,
            trainer.device,
            configuration,
        )

        save_pro(pro, args.filename)
    else:
        print(f'unknown command: {args.command}')


if __name__ == '__main__':
    main()
