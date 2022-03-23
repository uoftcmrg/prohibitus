from argparse import ArgumentParser

from torch import cat, long, no_grad, tensor

from prohibitus import ABCConfiguration, ABCDataset, ABCModel, ABCTrainer


@no_grad()
def infer(model, context, count, device, configuration):
    x = tensor(
        tuple(map(ord, context)),
        dtype=long,
    ).unsqueeze(0).to(device)

    model.eval()

    for _ in range(count):
        input_ = x[:, -configuration.chunk_size:]
        probabilities = model(input_)[:, -1, :]
        y = probabilities.multinomial(1)
        x = cat((x, y), 1)

    completion = ''.join(map(chr, x[0]))

    return completion


def main():
    parser = ArgumentParser(
        description='Train or infer music generation in abc format',
    )
    parser.add_argument('command', metavar='<command>')
    parser.add_argument('--filename', metavar='<filename to continue>')
    parser.add_argument(
        '--count',
        type=int,
        metavar='<number of generated characters>',
    )

    args = parser.parse_args()

    configuration = ABCConfiguration()
    model = ABCModel(configuration)
    train_dataset = ABCDataset(True, configuration)
    test_dataset = None
    trainer = ABCTrainer(model, train_dataset, test_dataset, configuration)

    if args.command == 'train':
        trainer.train()
    elif args.command == 'infer':
        with open(args.filename, encoding='utf-8') as file:
            content = file.read()

        content = infer(
            model,
            content,
            args.count,
            trainer.device,
            configuration,
        )

        with open(args.filename, 'w', encoding='utf-8') as file:
            file.write(content)
    else:
        print(f'unknown command: {args.command}')


if __name__ == '__main__':
    main()
