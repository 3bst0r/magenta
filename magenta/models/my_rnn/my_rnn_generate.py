import note_seq
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import magenta.common.beam_search

from magenta.models.my_rnn.my_simple_rnn_model import BASIC_DEFAULT_MIN_NOTE, BASIC_DEFAULT_MAX_NOTE


def one_hot_event(len, index):
    result = [0. for _ in range(len)]
    result[index] = 1.
    return result


def np_one_hot_event(len, index):
    result = np.zeros(len)
    result[index] = 1
    return result


# Returns a new np.array filled with 0., but predicted_event[np.argmax(predicted_event)] == 1.
def convert_to_one_hot(predicted_event):
    result = np.copy(predicted_event)
    index = np.argmax(result)
    result.fill(0.)
    result[index] = 1.
    return result


# returns the indices of the hot values in the supplied array
def convert_to_note_events(one_hot_events):
    _, result = np.where(one_hot_events == 1.)
    encoder_decoder = note_seq.MelodyOneHotEncoding(BASIC_DEFAULT_MIN_NOTE, BASIC_DEFAULT_MAX_NOTE)
    result = np.array([encoder_decoder.decode_event(event) for event in result])
    return result


def melody_seq_to_midi(event_seq, midi_file_path, qpm):
    note_events = convert_to_note_events(event_seq)
    output_sequence = note_seq.Melody(note_events).to_sequence(qpm=qpm)
    note_seq.midi_io.note_sequence_to_midi_file(output_sequence, midi_file_path)
    print("wrote midi output to {}".format(midi_file_path))


"""
Generates melodies step by step. At each step the most likely event is chosen.
"""


def generate_greedy(seed_seq, model, n):
    # model statefulness has to be true so LSTM layers keeps state between predictions. mimics one long sequence,
    # enables feeding output back into input for variable sequence length generation
    # batch input shape must be specified
    # reset state before sequence generation. previous sequence should not influence this sequence
    # TODO a config should keep track of where LSTM layers are
    assert model.layers[0].stateful
    assert model.layers[1].stateful
    model.reset_states()

    model.layers[0].return_sequences = True
    model.layers[1].return_sequences = True
    # first generate prediction for seed sequence one by one
    Y = feed_seq(model, seed_seq)

    # from here on, we feed predictions back as input
    Y = convert_to_one_hot(Y.flatten())
    yield Y
    X = Y.reshape(1, 1, Y.shape[0])
    for i in range(n - 1):
        Y = model.predict(X)
        Y = convert_to_one_hot(Y.flatten())
        yield Y
        X = Y.reshape(1, 1, Y.shape[0])


def get_lstm_state(model, lstm_layer_indices):
    K = tf.keras.backend
    lstm_state = []
    for i, lstm_layer_ind in enumerate(lstm_layer_indices):
        h = K.get_value(model.layers[lstm_layer_ind].states[0])
        c = K.get_value(model.layers[lstm_layer_ind].states[1])
        lstm_state.append((h, c))
    return lstm_state


def set_lstm_state(model, lstm_layer_indices, lstm_state):
    K = tf.keras.backend
    for i, lstm_layer_ind in enumerate(lstm_layer_indices):
        K.set_value(model.layers[lstm_layer_ind].states[0], lstm_state[i][0])  # h value
        K.set_value(model.layers[lstm_layer_ind].states[1], lstm_state[i][1])  # c value


"""
Feeds the model a seq, simply for setting up the internal state of the network.
"""


def feed_seq(model, seq):
    # seq = seq.reshape(seq.shape[0], seq.shape[1])
    Y = None
    for X in seq:
        X = X.reshape(1, 1, X.shape[0])
        Y = model.predict(X)
    return Y


def predict_from_state(model, X, state, lstm_layer_indices):
    """
    Sets the LSTM state, predicts from X and returns the new state.
    The internal state of the model gets changed
    :param model: LSTM model
    :param X: input to model
    :param state: model state
    :param lstm_layer_indices: which of the layers in the network are LSTM layers
    :return: (Y, state_new)
    """
    set_lstm_state(model=model, lstm_layer_indices=lstm_layer_indices, lstm_state=state)
    Y = model.predict(X)
    state_new = get_lstm_state(model=model, lstm_layer_indices=lstm_layer_indices)
    return Y, state_new


def generate_beam_search(seed_seq, model, lstm_layer_indices, n, beam_size=3, branch_factor=3):
    def generate_step_fn(sequences, states, scores):
        new_sequences = []
        new_states = []
        new_scores = []
        for i, (sequence, state, score) in enumerate(zip(sequences, states, scores)):
            X = sequence[-1, :].reshape(1, 1, sequence.shape[1])
            Y, new_state = predict_from_state(model, X, state,
                                              lstm_layer_indices)
            Y = Y.flatten()
            # add new sequences, states and scores for all predictions
            for i in range(len(Y)):
                new_sequence = np.concatenate((sequence, [one_hot_event(len(Y), i)]))
                new_sequences.append(new_sequence)
                new_states.append(new_state)
                new_scores.append(score - np.log(Y[i]))
        return new_sequences, new_states, new_scores

    feed_seq(model, seed_seq[:-1])
    initial_state = get_lstm_state(model, lstm_layer_indices)

    best_seq, state, loglik = magenta.common.beam_search(initial_sequence=seed_seq,
                                                         initial_state=initial_state,
                                                         generate_step_fn=generate_step_fn,
                                                         num_steps=n,
                                                         beam_size=beam_size,
                                                         branch_factor=branch_factor,
                                                         steps_per_iteration=1)
    return best_seq


def legacy_generate_beam_search(seed_seq, model, n, b=3, lstm_layer_indices=None):
    """
    Generates sequences using beam search algorithm
    :param seed_seq: the seed sequence
    :param model: LSTM model
    :param n: number of predictions requested
    :param b: beam width of beam search algorithm
    :param lstm_layer_indices: which of the layers in the network are LSTM layers
    """
    assert b > 0
    assert lstm_layer_indices is not None
    for ind in lstm_layer_indices:
        assert model.layers[ind].stateful
        model.layers[ind].return_sequences = True
    model.reset_states()

    event_size = seed_seq.shape[1]

    # run the seed sequence through
    Y = feed_seq(model=model, seq=seed_seq)

    # first guesses
    guesses = np.argsort(-Y.flatten())[:b]

    # we store a list of tuples of subsequent event indices as candidates
    candidate_inds = [(guess,) for guess in guesses]

    lstm_states = dict()
    # the empty tuple gives the starting state
    lstm_states[()] = get_lstm_state(model=model, lstm_layer_indices=lstm_layer_indices)
    # start the prediction loop with beam search
    for _ in range(n - 1):

        new_candidate_inds = []
        new_candidate_probs = np.array([])

        for candidate_ind in candidate_inds:
            candidate_ind = tuple(candidate_ind)
            # start with the last predicted id
            candidate = candidate_ind[-1]
            X = np_one_hot_event(event_size, candidate)
            X = X.reshape(1, 1, X.shape[0])

            (Y, new_model_state) = predict_from_state(model=model, X=X,
                                                      state=lstm_states[candidate_ind[:-1]],
                                                      lstm_layer_indices=lstm_layer_indices)
            # store the new state
            lstm_states[candidate_ind] = new_model_state
            Y = Y.flatten()
            top_predicted_inds = np.argsort(-Y)[:b]
            top_predicted_probs = Y[top_predicted_inds]
            # create new tuples in new_candidate_inds
            new_tuples = [candidate_ind + (pred,) for pred in top_predicted_inds]
            new_candidate_inds = new_candidate_inds + new_tuples
            new_candidate_probs = np.concatenate((new_candidate_probs, top_predicted_probs))

        # sort the predictions of this step, keep only the B best ones
        mask = np.argsort(-new_candidate_probs)[:b]
        # create a new dict so the unneeded LSTM states can be collected by gc
        new_lstm_states = dict()
        # TODO fix this saving lstm states logic: keep only b best ones
        for c in new_candidate_inds:
            one_before = tuple(c[:-1])
            new_lstm_states[one_before] = lstm_states[one_before]
        lstm_states = new_lstm_states
        candidate_inds = (np.array(new_candidate_inds, dtype=object))[mask]
    return np.array([np_one_hot_event(event_size, index) for index in candidate_inds[0]])


def plot_likelihoods_fn(model, event_size):
    def likelihood_for_events(x, y):
        z = np.zeros(x.shape)
        for i, note_event in enumerate(x[0]):
            X = np_one_hot_event(event_size, note_event)
            X = X.reshape((1, 1, event_size))
            z[i] = model.predict(X)
        return z

    x = np.arange(event_size)
    y = np.arange(event_size)
    X, Y = np.meshgrid(x, y)
    Z = likelihood_for_events(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='BrBG')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
