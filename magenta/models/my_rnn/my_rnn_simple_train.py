# %%
import sys

import tensorflow as tf
from .my_rnn_model import BASIC_EVENT_DIM
from .my_rnn_model import get_simple_rnn_model

print("executing eagerly:")
print(tf.executing_eagerly())

BATCH_SIZE = 128


def get_parse_function_shift(event_dim, label_shape=None):
    sequence_features = {
        'inputs': tf.io.FixedLenSequenceFeature(shape=[event_dim],
                                                dtype=tf.float32),
        'labels': tf.io.FixedLenSequenceFeature(shape=label_shape or [],
                                                dtype=tf.int64)}

    def shift_melody(example):
        # one example is one melody
        _, sequence = tf.parse_single_sequence_example(serialized=example, sequence_features=sequence_features)
        # return melody from first step as input and melody shifted by one step to the left as label
        return sequence['inputs'][:-1], sequence['inputs'][1:]

    return shift_melody


sequence_example_file_paths = [
    '/home/johannes/Documents/uni/msc/data/sequenceexamples/lmd_matched_A_basic/training_melodies.tfrecord']

# read data
ds = tf.data.TFRecordDataset(sequence_example_file_paths)

ds = ds.map(get_parse_function_shift(BASIC_EVENT_DIM))

ds = ds.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([None, BASIC_EVENT_DIM], [None, BASIC_EVENT_DIM]))
# shape is now [2             : input and label sequences
#               128           : batch size
#               ?             : padded sequence length per batch
#               38]           : event dimensionality

model = get_simple_rnn_model(BASIC_EVENT_DIM, is_Training=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(ds, epochs=50)
model.save('/tmp/simple_rnn_matched_A.model')
