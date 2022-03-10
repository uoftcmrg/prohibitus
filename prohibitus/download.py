from argparse import ArgumentParser
from csv import DictReader
from os import makedirs
from os.path import exists, isdir, join
from time import sleep, time
from urllib.parse import urljoin

from requests import get
from tqdm import tqdm


def download(source, destination, gateway, request_delay, chunk_size):
    rows = []

    with open(source, 'r') as file:
        reader = DictReader(file)

        for row in reader:
            rows.append(row)

    if not isdir(destination):
        makedirs(destination)

    for row in tqdm(rows):
        filename = join(destination, row['id'] + '.mscz')

        if exists(filename):
            continue

        url = urljoin(gateway, row['ref'])

        begin_time = time()

        with get(url, stream=True) as request:
            request.raise_for_status()

            with open(filename, 'wb') as file:
                for chunk in request.iter_content(chunk_size=chunk_size):
                    file.write(chunk)

        end_time = time()
        duration = end_time - begin_time

        if duration < request_delay:
            sleep(request_delay - duration)


def main():
    parser = ArgumentParser(
        description='Download musescore files from manifest',
    )
    parser.add_argument('source', metavar='<source filename>')
    parser.add_argument('destination', metavar='<destination dirname>')
    parser.add_argument(
        '--gateway',
        metavar='<IPFS HTTP Gateway>',
        default='https://ipfs.infura.io/',
    )
    parser.add_argument(
        '--request_delay',
        type=float,
        metavar='<IPFS HTTP request delay>',
        default=0.6,
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        metavar='<download chunk size>',
        default=8192,
    )

    args = parser.parse_args()

    download(
        args.source,
        args.destination,
        args.gateway,
        args.request_delay,
        args.chunk_size,
    )


if __name__ == '__main__':
    main()
