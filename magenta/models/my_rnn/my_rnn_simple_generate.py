import argparse
import os

import numpy as np
from tensorflow.compat.v1 import keras

from magenta.common import is_valid_file
from magenta.models.my_rnn.my_rnn_generate import generate_greedy
from magenta.models.my_rnn.my_rnn_generate import one_hot_event, melody_seq_to_midi, generate_beam_search
from magenta.models.my_rnn.my_rnn_generate import plot_likelihoods_fn
from magenta.models.my_rnn.my_simple_rnn_model import BASIC_EVENT_DIM
from magenta.models.my_rnn.my_simple_rnn_model import get_simple_rnn_model

tmp_dir = os.environ["TMPDIR"]


def main(trained_model_file_path, midi_output_file_path, temperature, beam_search=False, plot_likelihoods=False):
    saved_model = keras.models.load_model(trained_model_file_path)

    model = get_simple_rnn_model(event_dim=BASIC_EVENT_DIM, is_Training=False, temperature=temperature)
    # set weights of saved pre trained model
    for saved_layer, model_layer in zip(saved_model.layers, model.layers):
        model_layer.set_weights(saved_layer.get_weights())

    # create example seed
    X = [one_hot_event(BASIC_EVENT_DIM, i) for i in [25]]
    X = np.array(X)

    if plot_likelihoods:
        print("plot likelihoods")
        plot_likelihoods_fn(model, BASIC_EVENT_DIM)

    melody_seq = None

    if beam_search:
        print("beam search")
        melody_seq = generate_beam_search(seed_seq=X, model=model, n=64, beam_size=1, branch_factor=1,
                                          lstm_layer_indices=[0, 1])
        for note_event in melody_seq:
            print(np.where(note_event == 1))
    else:
        print("greedy")
        for note_event in generate_greedy(seed_seq=X, model=model, n=64):
            print(np.where(note_event == 1))

    if midi_output_file_path is not None:
        melody_seq_to_midi(melody_seq, midi_output_file_path, 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate from RNN Model")
    parser.add_argument('--trained_model_file', dest="filename", required=True,
                        help=".tf file containing the trained weights",
                        type=lambda x: is_valid_file(parser, os.path.join(tmp_dir, x)))
    parser.add_argument('--output_midi_file', dest="output_midi_file", required=False,
                        help="filename of midi output file")
    parser.add_argument('--beam_search', dest="beam_search", required=False,
                        help="Whether to use beam search for generation. Greedy generation will be used otherwise.",
                        action='store_true')
    parser.add_argument('--plot_likelihoods', dest="plot_likelihoods", required=False,
                        help="Plot likelihoods of generated values",
                        action='store_true')
    parser.add_argument('--temperature', dest="temperature", required=False,
                        help='Temperature of softmax for generation',
                        default=1.0)
    args = parser.parse_args()

    if args.output_midi_file is not None:
        args.output_midi_file = os.path.join(tmp_dir, args.output_midi_file)
    main(trained_model_file_path=args.filename, midi_output_file_path=args.output_midi_file,
         beam_search=args.beam_search, plot_likelihoods=args.plot_likelihoods, temperature=float(args.temperature))
