import numpy as np


def one_hot_event(len, index):
    result = [0. for _ in range(len)]
    result[index] = 1.
    return result


# Returns a new np.array filled with 0., but predicted_event[np.argmax(predicted_event)] == 1.
def convert_to_one_hot(predicted_event):
    result = np.copy(predicted_event)
    index = np.argmax(result)
    result.fill(0.)
    result[index] = 1.
    return result


def generate(seed_seq, model, n):
    # model statefulness has to be true so LSTM layers keeps state between predictions. mimics one long sequence,
    # enables feeding output back into input for variable sequence length generation
    # batch input shape must be specified
    # reset state before sequence generation. previous sequence should not influence this sequence
    assert model.layers[0].stateful
    model.reset_states()

    model.layers[0].return_sequences = True
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