""" This module prepares midi file data and feeds it to the neural
    network for training """

from keras.callbacks import ModelCheckpoint

import utils


encoding_file = 'encoding'


def train_network():
    """ Train a Neural Network to generate music """
    notes = utils.get_notes_duration()
    network_input, _, network_output = utils.prepare_sequences(notes)
    model = utils.create_network(network_input)
    train(model, network_input, network_output)


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output,
              epochs=200, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
