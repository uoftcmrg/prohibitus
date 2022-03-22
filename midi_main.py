from argparse import ArgumentParser

from torch import empty, multinomial, no_grad

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
    x = empty(1, len(context) + count).to(device)
    x[0, :len(context)] = context

    model.eval()

    for i in range(len(context), len(context) + count):
        input_ = x[:, max(0, i - configuration.chunk_size):i]
        probabilities = model(input_)[:, -1, :]
        y = multinomial(probabilities, num_samples=1)
        x[:, i:i + 1] = y

    return x.tolist()


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
        pro = load_pro(args.filename, configuration)

        pro = infer(
            model,
            pro,
            args.count,
            trainer.device,
            configuration,
        )

        save_pro(pro, args.filename, configuration)
    else:
        print(f'unknown command: {args.command}')


if __name__ == '__main__':
    main()
