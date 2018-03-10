import glob
import json
from collections import namedtuple

import numpy
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from music21 import converter, instrument, note, chord

encoding_file = 'encoding'
sequence_length = 100
seed_file = 'midi_songs/8.mid'

Note = namedtuple('Note', ['pitch', 'duration', 'offset'])


def nr_pitches():
    """ Return the number of original pitches """
    return len(get_note_to_int())


def get_note_to_int():
    with open(encoding_file) as fhandle:
        return {key: float(val) for key, val in json.load(fhandle).items()}


def get_int_to_note():
    return {val: key for key, val in get_note_to_int().items()}


def encode_duration(dur):
    if dur < 0.5:
        return 0
    if dur < 1:
        return 1
    if dur < 2:
        return 2
    if dur < 4:
        return 3
    if dur < 8:
        return 4
    return 5


def decode_duration(dur_encoded: int):
    if dur_encoded == 0:
        return 0.0
    return float(2) ** (dur_encoded - 2)


def encode_offset(dur):
    dur = dur * 12
    dur = min(24, max(dur, 0))
    return int(dur)


def decode_offset(dur):
    return float(dur) / 12.0


def get_notes_duration():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """

    pitches = []
    durations = []
    offsets = []
    last_offset = 0
    for file in glob.glob("midi_songs/8.mid"):
        midi = converter.parse(file)

        parts = instrument.partitionByInstrument(midi)

        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                pitches.append(str(element.pitch))
                durations.append(encode_duration(element.duration.quarterLength))
                offset = element.offset
                print(encode_offset(offset - last_offset))
                offsets.append(encode_offset(offset - last_offset))
                last_offset = offset
            elif isinstance(element, chord.Chord):
                pitches.append('.'.join(str(n) for n in element.normalOrder))
                durations.append(encode_duration(element.duration.quarterLength))
                offset = element.offset
                print(encode_offset(offset - last_offset))
                offsets.append(encode_offset(offset - last_offset))
                last_offset = offset

    note_to_int = {pitch: float(i) for i, pitch in enumerate(set(pitches))}

    with open(encoding_file, 'w') as fhandle:
        json.dump(note_to_int, fhandle)

    pitches = [float(note_to_int[pitch]) for pitch in pitches]

    notes = list(map(Note, pitches, durations, offsets))
    return notes


def get_nodes_duration_for_prediction(fname):

    pitches = []
    durations = []
    offsets = []
    last_offset = 0
    midi = converter.parse(fname)

    parts = instrument.partitionByInstrument(midi)

    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            pitches.append(str(element.pitch))
            durations.append(encode_duration(element.duration.quarterLength))
            offset = element.offset
            offsets.append(encode_offset(offset - last_offset))
            last_offset = offset
        elif isinstance(element, chord.Chord):
            pitches.append('.'.join(str(n) for n in element.normalOrder))
            durations.append(encode_duration(element.duration.quarterLength))
            offset = element.offset
            offsets.append(encode_offset(offset - last_offset))
            last_offset = offset

    note_to_int = get_note_to_int()
    pitches = [float(note_to_int[pitch]) for pitch in pitches]

    notes = list(map(Note, pitches, durations, offsets))
    return notes


def prepare_sequences(notes):
    """ Prepare the sequences used by the Neural Network """
    network_input = []
    pitch_output = []
    duration_output = []
    offset_output = []
    nr_features = len(notes[0])
    n_vocab = nr_pitches()

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        network_input.append(notes[i:i + sequence_length])
        pitch_output.append(notes[i + sequence_length].pitch)
        duration_output.append(notes[i + sequence_length].duration)
        offset_output.append(notes[i + sequence_length].offset)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length,
                                                     nr_features))
    # normalize input
    normalized_input[:, :, 0] = normalized_input[:, :, 0] / float(n_vocab)
    normalized_input[:, :, 1] = normalized_input[:, :, 1] / 5.0
    normalized_input[:, :, 2] = normalized_input[:, :, 2] / 24.0
    pitch_output = to_categorical(pitch_output, num_classes=n_vocab)
    duration_output = to_categorical(duration_output, num_classes=5)
    offset_output = to_categorical(offset_output, num_classes=25)
    return normalized_input, network_input, [pitch_output, duration_output, offset_output]


def create_network(network_input):
    """ create the structure of the neural network """
    n_vocab = nr_pitches()
    input = Input(shape=(network_input.shape[1], network_input.shape[2]))
    lstm1 = LSTM(
        128,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    )(input)
    d1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(128, return_sequences=True)(d1)
    d2 = Dropout(0.3)(lstm2)
    lstm3 = LSTM(128)(d2)
    dense = Dense(64)(lstm3)
    d3 = Dropout(0.3)(dense)
    out_pitch = Dense(n_vocab, activation='softmax')(d3)
    out_duration = Dense(5, activation='softmax')(d3)
    out_offset = Dense(25, activation='softmax')(d3)

    model = Model(inputs=[input], outputs=[out_pitch, out_duration, out_offset])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


if __name__ == '__main__':
    notes = get_notes_duration()
    inp, _, outp = prepare_sequences(notes)
