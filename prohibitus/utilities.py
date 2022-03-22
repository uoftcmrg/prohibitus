from functools import cache, partial
from itertools import filterfalse
from operator import getitem

from pretty_midi import Instrument, Note, PrettyMIDI


def seconds_to_index(time, configuration):
    return round((time * 1000) ** (1 / configuration.time_power))


def index_to_seconds(index, configuration):
    return (index ** configuration.time_power) / 1000


def note_to_semipro(note, time, configuration):
    index = seconds_to_index(note.start - time, configuration)
    delay = configuration.delays[min(index, len(configuration.delays) - 1)]

    pitch = configuration.pitches[note.pitch]
    velocity = configuration.velocities[note.velocity]

    index = seconds_to_index(note.duration, configuration)
    duration = \
        configuration.durations[min(index, len(configuration.durations) - 1)]

    tokens = configuration.null, delay, pitch, velocity, duration

    return tokens


def semipro_to_note(semipro, time, configuration):
    _, delay, pitch, velocity, duration = semipro

    index = configuration.delays.index(delay)
    delay = index_to_seconds(index, configuration)

    pitch = configuration.pitches.index(pitch)
    velocity = configuration.velocities.index(velocity)

    index = configuration.durations.index(duration)
    duration = index_to_seconds(index, configuration)

    note = Note(velocity, pitch, time + delay, time + delay + duration)

    return note


@cache
def load_pro(midi_file, configuration):
    try:
        pm = PrettyMIDI(midi_file)
    except Exception:
        return []

    notes = []

    for instrument in pm.instruments:
        notes.extend(instrument.notes)

    notes.sort(
        key=lambda note: (note.start, note.pitch, note.velocity, note.end),
    )

    tokens = []
    time = notes[0].start if notes else 0

    for note in notes:
        semipro = note_to_semipro(note, time, configuration)

        tokens.extend(semipro)
        time = note.start

    return tokens


def save_pro(pro, filename, configuration):
    notes = []
    time = 0
    indices = tuple(filterfalse(partial(getitem, pro), range(len(pro))))

    for i, (begin, end) in enumerate(zip(indices, indices[1:] + (len(pro),))):
        semipro = pro[begin:end]

        try:
            note = semipro_to_note(semipro, time, configuration)
        except Exception:
            print(f'Invalid syntax at Line {i}: {semipro}')
            continue

        notes.append(note)
        time = note.start

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
