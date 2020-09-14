from tensorflow import keras
from tensorflow.keras import layers

BASIC_DEFAULT_MIN_NOTE = 48
BASIC_DEFAULT_MAX_NOTE = 84
# dimensionality of one event
BASIC_EVENT_DIM = BASIC_DEFAULT_MAX_NOTE - BASIC_DEFAULT_MIN_NOTE + 2  # all note on events, note off, rest

def get_simple_rnn_model(event_dim, is_Training):
    model = keras.Sequential()
    # input_shape: (None,         : different sequence lengths (per batch; every sequence in one batch does have the same dimension)
    #               EVENT_DIM)    : dimensionality of one event
    lstm_args = {'units': 50,
            'input_shape': (None, event_dim),
            'return_sequences': True,
            }
    # for generating
    if not is_Training:
        # we predict one by one event
        lstm_args['input_shape'] = (1, event_dim)
        lstm_args['batch_input_shape'] = (1, 1, event_dim)
        lstm_args['stateful'] = True

    model.add(layers.LSTM(**lstm_args))
    model.add(layers.Dense(units=event_dim, activation='softmax'))

    return model
