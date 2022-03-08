"""Copyright (c) 2022 - Juho Kim

A program to convert mscz files to abc files.

MuseScore3.exe path must be provided as an argument or in %PATH%.
"""
from argparse import ArgumentParser
from csv import DictReader
from glob import glob
from json import dump
from os import makedirs, remove, system
from os.path import abspath, exists, isdir, join
from time import sleep, time
from urllib.parse import urljoin

from requests import get
from tqdm import tqdm


def convert(source, destination, musescore_path):
    source = abspath(source) + '/'
    destination = abspath(destination) + '/'

    job = []
    filenames = glob(join(source, '**/*.mscz'), recursive=True)

    for filename in filenames:
        job.append(
            {
                'in': filename,
                'out': join(
                    destination,
                    filename[len(source):].replace('.mscz', '.xml'),
                ),
            },
        )

    job_path = join(destination, 'job.json')

    if not isdir(destination):
        makedirs(destination)

    with open(job_path, 'w') as file:
        dump(job, file)

    system(f'"{musescore_path}" -j {job_path}')

    remove(job_path)

    exit()

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
    parser.add_argument('source', metavar='<source dirname>')
    parser.add_argument('destination', metavar='<destination dirname>')
    parser.add_argument(
        '--musescore_path',
        default='MuseScore3.exe',
        metavar='<MuseScore Path>',
    )

    args = parser.parse_args()

    convert(args.source, args.destination, args.musescore_path)


if __name__ == '__main__':
    main()
