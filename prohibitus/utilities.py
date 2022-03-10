import numpy as np
from pretty_midi import Instrument, Note, PrettyMIDI


def load_piano_roll(midi_file, configuration):
    threshold = configuration.threshold * 128

    pm = PrettyMIDI(midi_file)
    piano_roll = pm.get_piano_roll(configuration.framerate).T

    for instrument in pm.instruments:
        for note in instrument.notes:
            i = int(note.start * configuration.framerate)

            if i != 0 and piano_roll[i, note.pitch] <= threshold \
                    + piano_roll[i - 1, note.pitch]:
                piano_roll[i - 1, note.pitch] = 0

    return (piano_roll / 128).clip(0, 1)


def save_piano_roll(piano_roll, filename, configuration):
    piano_roll = (piano_roll * 128).clip(0, 127)
    threshold = configuration.threshold * 128

    pm = PrettyMIDI()
    instrument = Instrument(0)

    piano_roll = np.pad(piano_roll, ((1, 1), (0, 0)))
    time_count, note_count = piano_roll.shape
    indices = np.nonzero(np.diff(piano_roll, axis=0))

    velocities = np.zeros(note_count, dtype=int)
    times = np.zeros(note_count)

    for f, n in zip(*indices):
        velocity = piano_roll[f + 1, n]
        time = f / configuration.framerate

        if velocity > threshold + velocities[n] or velocity < threshold:
            note = Note(velocities[n], n, times[n], time)
            instrument.notes.append(note)
            velocities[n] = 0

        velocities[n] = velocity
        times[n] = time

    pm.instruments.append(instrument)
    pm.write(filename)
