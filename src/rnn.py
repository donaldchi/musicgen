from music21 import converter, instrument, note, chord
import glob
import pickle
import numpy as np
from keras.utils import np_utils
import tensorflow as tf


class PrepareData:

    def __init__(self):
        pass

    def get_notes(self, file_path):
        notes = []
        for file in glob.glob(file_path):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

            with open('data/notes', 'wb') as file_path_notes:
                pickle.dump(notes, file_path_notes)
        return notes

    def create_data(self, notes, pitchnames, n_vocab):
        """ Prepare the sequences used by the Neural Network """
        sequence_length = 100

        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        network_input = []
        network_output = []

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)

        network_output = np_utils.to_categorical(network_output)

        return (network_input, network_output)


class RNN:
    def __init__(self, n_vocab):
        self.inpute_layer_size = 100
        self.hidden_lstm_layer_size = 512
        self.dropout_layer1_size = 512
        self.dense_layer1_size = 256
        self.dropout_layer2_size = 256
        self.dense_layer2_size = n_vocab

    def inference(self, input, initial_state):



if __name__=="__main__":
    # test_rnn = RNN()
    # greate notes from midi files
    file_path = "../tutorial/Classical-Piano-Composer/midi_songs/*.mid"
    test_data = PrepareData()
    notes = test_data.get_notes(file_path)

    # get amount of  picthnames
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    # get data
    network_input, network_output = test_data.create_data(notes, pitchnames, n_vocab)
