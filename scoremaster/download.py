""" A program to download musescore files and convert it into abc files.

Credit to Xmader for the dataset: https://github.com/Xmader/musescore-dataset
"""
from argparse import ArgumentParser
from csv import DictReader
from os import mkdir
from os.path import exists, isdir, join
from time import sleep
from urllib.parse import urljoin

from requests import get
from tqdm import tqdm


def download(source, destination, gateway, chunk_size):
    rows = []

    with open(source, 'r') as file:
        reader = DictReader(file)

        for row in reader:
            rows.append(row)

    if not isdir(destination):
        mkdir(destination)

    for row in tqdm(rows):
        filename = join(destination, row['id'] + '.msz')

        if exists(filename):
            continue

        url = urljoin(gateway, row['ref'])

        with get(url, stream=True) as request:
            request.raise_for_status()

            with open(filename, 'wb') as file:
                for chunk in request.iter_content(chunk_size=chunk_size):
                    file.write(chunk)

        sleep(0.3)


def main():
    parser = ArgumentParser(
        description='Download musescore files from manifest',
    )
    parser.add_argument('source', metavar='<source filename>')
    parser.add_argument('destination', metavar='<destination dirname>')
    parser.add_argument(
        '--chunk_size',
        type=int,
        metavar='<download chunk size>',
        default=8192,
    )
    parser.add_argument(
        '--request_delay',
        type=float,
        metavar='<IPFS HTTP request delay>',
        default=0.3,
    )
    parser.add_argument(
        '--gateway',
        metavar='<IPFS HTTP Gateway>',
        default='https://ipfs.infura.io/',
    )

    args = parser.parse_args()

    download(args.source, args.destination, args.gateway, args.chunk_size)


if __name__ == '__main__':
    main()
