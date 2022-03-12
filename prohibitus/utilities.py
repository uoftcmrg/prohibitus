import numpy as np
from pretty_midi import Instrument, Note, PrettyMIDI


def int_to_reversed_str(value):
    return ''.join(reversed(str(value)))


def reversed_str_to_int(value):
    return int(''.join(reversed(value)))


def seconds_to_reversed_str_milliseconds(value):
    return int_to_reversed_str(int(value * 1000))


def reversed_str_milliseconds_to_seconds(value):
    return reversed_str_to_int(value) / 1000


def load_pro(midi_file):
    try:
        pm = PrettyMIDI(midi_file)
    except Exception:
        return ''

    notes = []

    for instrument in pm.instruments:
        notes.extend(instrument.notes)

    notes.sort(
        key=lambda note: (note.start, note.pitch, note.velocity, note.end),
    )

    lines = []
    time = notes[0].start if notes else 0

    for note in notes:
        delay = seconds_to_reversed_str_milliseconds(note.start - time)
        pitch = int_to_reversed_str(note.pitch)
        velocity = int_to_reversed_str(note.velocity)
        duration = seconds_to_reversed_str_milliseconds(note.duration)

        line = f'{delay} {pitch} {velocity} {duration}'

        lines.append(line)
        time = note.start

    return '\n'.join(lines)


def save_pro(pro, filename):
    notes = []
    time = 0

    for line in pro.split('\n'):
        try:
            delay, pitch, velocity, duration = line.split()

            delay = reversed_str_milliseconds_to_seconds(delay)
            pitch = reversed_str_to_int(pitch)
            velocity = reversed_str_to_int(velocity)
            duration = reversed_str_milliseconds_to_seconds(duration)
        except ValueError:
            print(f'Invalid syntax: {line}')
        else:
            pitch = np.clip(pitch, 0, 127)
            velocity = np.clip(velocity, 0, 127)
            time += delay

            note = Note(velocity, pitch, time, time + duration)

            notes.append(note)

    pm = PrettyMIDI()
    instrument = Instrument(0)
    instrument.notes = notes

    pm.instruments.append(instrument)
    pm.write(filename)


def trim_midi(midi_file, start, end):
    pm1 = PrettyMIDI(midi_file)
    pm2 = PrettyMIDI()

    for instrument1 in pm1.instruments:
        instrument2 = Instrument(
            instrument1.program,
            instrument1.is_drum,
            instrument1.name,
        )

        for note1 in instrument1.notes:
            if start <= note1.end <= end or start <= note1.start <= end:
                note2 = Note(
                    note1.velocity,
                    note1.pitch,
                    max(note1.start, start),
                    min(note1.end, end),
                )

                instrument2.notes.append(note2)

        pm2.instruments.append(instrument2)

    pm2.write(midi_file)
