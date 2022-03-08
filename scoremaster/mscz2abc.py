"""Copyright (c) 2022 - Juho Kim

A program to convert mscz files to abc files.
"""
from argparse import ArgumentParser
from csv import DictReader
from os import makedirs
from os.path import exists, isdir, join
from time import sleep, time
from urllib.parse import urljoin

from requests import get
from tqdm import tqdm


def convert(source, destination):
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
        description='Convert all mscz files from source to destination',
    )
    parser.add_argument('source', metavar='<source file/dirname>')
    parser.add_argument('destination', metavar='<destination file/dirname>')

    args = parser.parse_args()

    convert(args.source, args.destination)


if __name__ == '__main__':
    main()
