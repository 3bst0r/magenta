import argparse
import numpy as np
from tensorflow import keras
from magenta.models.my_rnn.my_simple_rnn_model import get_simple_rnn_model
from magenta.models.my_rnn.my_simple_rnn_model import BASIC_EVENT_DIM
from magenta.models.my_rnn.my_rnn_generate import one_hot_event
from magenta.models.my_rnn.my_rnn_generate import generate


def main():
    saved_model = keras.models.load_model('/home/johannes/Documents/uni/msc/systems/trained-models/mymodels/twolayer_lstm')

    model = get_simple_rnn_model(BASIC_EVENT_DIM, is_Training=False)
    # set weights of saved pre trained model
    for saved_layer, model_layer in zip(saved_model.layers, model.layers):
        model_layer.set_weights(saved_layer.get_weights())

    # create example seed
    X = [one_hot_event(BASIC_EVENT_DIM, i) for i in [25, 27, 0, 27]]
    X = np.array(X)

    for note_event in generate(X, model, 64):
        print(np.where(note_event == 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate from RNN Model")
    args = parser.parse_args()
    main()
