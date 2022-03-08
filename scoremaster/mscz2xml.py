"""Copyright (c) 2022 - Juho Kim

A program to convert mscz files to abc files.

MuseScore3.exe path must be provided as an argument or in %PATH%.
"""
from argparse import ArgumentParser
from glob import glob
from json import dump
from os import makedirs, remove, system
from os.path import isdir, join


def convert(source, destination, musescore_path):
    if not isdir(destination):
        makedirs(destination)

    inputs = glob(join(source, '*.mscz'), recursive=True)
    outputs = []
    job = []

    for input_ in inputs:
        output = join(
            destination,
            input_[len(source):].replace('.mscz', '.xml'),
        )

        outputs.append(output)
        job.append({'in': input_, 'out': output})

    job_path = join(destination, 'job.json')

    with open(job_path, 'w') as file:
        dump(job, file)

    system(f'"{musescore_path}" -j {job_path}')

    remove(job_path)


def main():
    parser = ArgumentParser(
        description='Convert all mscz files from source to destination',
    )
    parser.add_argument('source', metavar='<source dirname>')
    parser.add_argument('destination', metavar='<destination dirname>')
    parser.add_argument(
        'musescore_path',
        metavar='<MuseScore3 executable path>',
    )

    args = parser.parse_args()

    convert(args.source, args.destination, args.musescore_path)


if __name__ == '__main__':
    main()
