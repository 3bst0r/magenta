from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers

BASIC_DEFAULT_MIN_NOTE = 48
BASIC_DEFAULT_MAX_NOTE = 84
# dimensionality of one event
BASIC_EVENT_DIM = BASIC_DEFAULT_MAX_NOTE - BASIC_DEFAULT_MIN_NOTE + 2  # all note on events, note off, rest
LOOKBACK_RNN_INPUT_EVENT_DIM = 120


def get_simple_rnn_model(event_dim, is_Training, temperature=1):
    # input_shape: (None,         : different sequence lengths (per batch; every sequence in one batch does have the same dimension)
    #               EVENT_DIM)    : dimensionality of one event
    layer_one_args = {'units': 128,
                      'input_shape': (None, event_dim),
                      'return_sequences': True,
                      'dropout': 0.5,
                      'recurrent_dropout': 0.5,
                      }
    layer_two_args = {'units': 128,
                      'return_sequences': True,
                      'dropout': 0.5,
                      'recurrent_dropout': 0.5,
                      }
    # for generating
    if not is_Training:
        # we predict one by one event
        layer_one_args['input_shape'] = (1, event_dim)
        layer_one_args['batch_input_shape'] = (1, 1, event_dim)
        layer_one_args['stateful'] = True
        layer_two_args['stateful'] = True

    model = keras.Sequential()
    model.add(layers.LSTM(**layer_one_args))
    # second LSTM layer
    model.add(layers.LSTM(**layer_two_args))
    model.add(layers.Lambda(lambda x: x/temperature))
    model.add(layers.Dense(units=event_dim, activation='softmax'))

    return model
