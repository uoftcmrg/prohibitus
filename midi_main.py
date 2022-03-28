from argparse import ArgumentParser

from torch import arange, empty, long, no_grad, tensor

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
    if not context:
        context = 0,

    masks = [
        tensor(configuration.null),
        tensor(configuration.delays),
        tensor(configuration.pitches),
        tensor(configuration.velocities),
        tensor(configuration.durations),
    ]
    index = None

    for i, mask in enumerate(masks):
        if context[-1] in mask:
            index = (i + 1) % len(masks)

        indices = arange(configuration.token_count)
        indices[mask] = 0
        masks[i] = indices

    x = empty(1, len(context) + count, dtype=long).to(device)
    x[0, :len(context)] = tensor(context)

    model.eval()

    for i in range(len(context), len(context) + count):
        input_ = x[:, max(0, i - configuration.chunk_size):i]
        logits = model(input_)[:, -1, :]

        if configuration.temperature is not None:
            logits /= configuration.temperature

        logits[masks[index]] = 0
        probabilities = logits.softmax(-1)
        y = probabilities.multinomial(1)

        index = (index + 1) % len(masks)
        x[:, i:i + 1] = y

    return x.tolist()[0]


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
