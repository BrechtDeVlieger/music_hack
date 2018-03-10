""" This module generates notes for a midi file using the
    trained neural network """
import numpy
import sys
from music21 import instrument, note, stream, chord, duration

import utils

if len(sys.argv) > 1:
    weights = sys.argv[1]
else:
    raise ValueError("Forgot to pass the model!")

encoding_file = 'encoding'


generate_from = 'midi_songs/ahead_on_our_way_piano.mid'
lstm_length = 100


def generate():
    """ Generate a piano midi file """

    # Get all pitch names
    notes = utils.get_nodes_duration_for_prediction(utils.seed_file)

    normalized_input, network_input, _ = utils.prepare_sequences(notes)
    model = utils.create_network(normalized_input)
    # Load the weights to each node
    model.load_weights(weights)

    prediction_output = generate_notes(model, network_input)
    create_midi(prediction_output)


def generate_notes(model, network_input):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    n_vocab = utils.nr_pitches()
    int_to_note = utils.get_int_to_note()
    start = numpy.random.randint(0, len(network_input)-1)

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 3))
        prediction = model.predict(prediction_input, verbose=0)

        pitch_pred, dur_pred, offset_pred = prediction
        pitch_index = numpy.argmax(pitch_pred)
        dur_index = numpy.argmax(dur_pred)
        offset_index = numpy.argmax(offset_pred)

        pitch = int_to_note[pitch_index]
        dur = utils.decode_duration(dur_index)
        offset = utils.decode_offset(offset_index)
        prediction_output.append([pitch, dur, offset])

        pattern.append((pitch_index/n_vocab, dur_index/5, offset/25))
        pattern = pattern[1:len(pattern)]
        print('{}: {}: {}'.format(pitch, dur, offset))

    return prediction_output


def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern, dur, offset in prediction_output:
        # pattern is a chord
        dur = duration.Duration(dur)
        last_offset = 0
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                new_note.duration = dur
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = dur
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = dur
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        last_offset += offset

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')


if __name__ == '__main__':
    generate()
