import numpy as np
import tensorflow as tf


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
    seed_seq = seed_seq.reshape((1, seed_seq.shape[0], seed_seq.shape[1]))
    Y = None
    for X in seed_seq[0]:
        X = [[[X]]]
        Y = model.predict(X)

    # from here on, we feed predictions back as input
    Y = convert_to_one_hot(Y.flatten())
    yield Y
    X = [[[Y]]]
    for i in range(n - 1):
        Y = model.predict(X)
        Y = convert_to_one_hot(Y.flatten())
        yield Y
        X = [[[Y]]]


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
    seq = seq.reshape((1, seq.shape[0], seq.shape[1]))
    Y = None
    for X in seq[0]:
        Y = model.predict([[[X]]])
    return Y


def predict_from_state(model, X, state, lstm_layer_indices):
    """
    Sets the LSTM state, predicts from X and returns the new state.
    The internal state of the model gets changed
    :param model: LSTM model
    :param X: input to model
    :param state: model state
    :param lstm_layer_indices: hich of the layers in the network are LSTM layers
    :return: (Y, state_new)
    """
    set_lstm_state(model=model, lstm_layer_indices=lstm_layer_indices, lstm_state=state)
    Y = model.predict(X)
    state_new = get_lstm_state(model=model, lstm_layer_indices=lstm_layer_indices)
    return (Y, state_new)


def generate_beam_search(seed_seq, model, n, b=3, lstm_layer_indices=None):
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
            X = [[[X]]]

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
        # TODO fix this saving lstm states logic: keep only B best ones
        for c in candidate_inds:
            new_lstm_states[tuple(c)] = lstm_states[tuple(c)]
        lstm_states = new_lstm_states
        candidate_inds = np.array(new_candidate_inds, dtype=object)[mask]
    # TODO return the final melody
    return list(candidate_inds[0])


def generate_beam_search_broken(seed_seq, model, n, beam_width=3, lstm_layer_indices=None):
    # model statefulness has to be true so LSTM layers keeps state between predictions. mimics one long sequence,
    # enables feeding output back into input for variable sequence length generation
    # batch input shape must be specified
    # reset state before sequence generation. previous sequence should not influence this sequence
    # TODO a config should keep track of where LSTM layers are
    if lstm_layer_indices is None:
        lstm_layer_indices = [0, 1]
    for ind in lstm_layer_indices:
        assert model.layers[ind].stateful
        model.layers[ind].return_sequences = True
    model.reset_states()

    event_size = seed_seq.shape[1]

    # first generate prediction for seed sequence one by one
    seed_seq = seed_seq.reshape((1, seed_seq.shape[0], seed_seq.shape[1]))
    Y = None
    for X in seed_seq[0]:
        X = [[[X]]]
        Y = model.predict(X)

    # from here on, we feed predictions back as input
    # beam search start

    Y = Y.flatten()
    candidate_inds = -np.argsort(Y)[:beam_width]
    # which of the beam_width choices will we output for the prev step

    max_probs_predicted_prev = np.empty(0)
    max_inds_predicted_prev = np.empty(0)

    for i in range(n - 1):

        # save state before trying multiple inputs
        lstm_state = get_lstm_state(model, lstm_layer_indices)
        # find most probable for step before
        prev_max_inds = np.empty(beam_width, dtype=np.int)
        prev_max_inds.fill(candidate_inds[0])
        for last_ind in candidate_inds:
            # input each into the network and calculate joint softmax probability
            X = [[[np_one_hot_event(event_size, last_ind)]]]
            Y = model.predict(X)
            Y = Y.flatten()
            max_inds_predicted = -np.argsort(Y)[:beam_width]
            max_probs_predicted = [Y[i] for i in max_inds_predicted]
            # compare with probs from before by concatenating and sorting
            max_probs_predicted_prev = np.concatenate((max_probs_predicted, max_probs_predicted_prev))
            max_inds_predicted_prev = np.concatenate((max_inds_predicted, max_inds_predicted_prev))
            sort_mask = -np.argsort(max_probs_predicted_prev)[:beam_width]
            max_probs_predicted_prev = [max_probs_predicted_prev[i] for i in sort_mask]
            max_inds_predicted_prev = [max_inds_predicted_prev[i] for i in sort_mask]
            last_inds = np.empty(beam_width, dtype=np.int)
            last_inds.fill(last_ind)
            prev_max_inds = np.concatenate((prev_max_inds, last_inds))
            prev_max_inds = [prev_max_inds[i] for i in sort_mask][:beam_width]
            # reset state to try out next input
            set_lstm_state(model, lstm_layer_indices, lstm_state)
        # the most probable events for the next step are our new candidates
        candidate_inds = prev_max_inds

        # we now yield the prediction that produced the most probable next step
        Y = np_one_hot_event(event_size, prev_max_inds[0])
        yield Y
        # run it through the LSTM again to set its state up for the next prediction
        model.predict([[[Y]]])
