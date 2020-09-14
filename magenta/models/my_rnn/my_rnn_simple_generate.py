import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from magenta.models.my_rnn.my_rnn_model import get_simple_rnn_model
from magenta.models.my_rnn.my_rnn_model import BASIC_EVENT_DIM
from magenta.models.my_rnn.my_rnn_generate import one_hot_event
from magenta.models.my_rnn.my_rnn_generate import generate


saved_model = keras.models.load_model('/home/johannes/Documents/uni/msc/systems/trained-models/mymodels/basic_lstm'
                                      '/simple_rnn_matched_A.model')

model = get_simple_rnn_model(BASIC_EVENT_DIM, is_Training=False)
# set weights of saved pre trained model
for saved_layer, model_layer in zip(saved_model.layers, model.layers):
    model_layer.set_weights(saved_layer.get_weights())

# create example seed
X = [one_hot_event(BASIC_EVENT_DIM, i) for i in np.full((20,), 1)]
X = np.array(X)

for note_event in generate(X, model, 64):
    print(np.where(note_event == 1))
