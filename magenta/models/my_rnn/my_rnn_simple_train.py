# %%
import sys
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from magenta.models.my_rnn.my_simple_rnn_model import BASIC_EVENT_DIM
from magenta.models.my_rnn.my_simple_rnn_model import get_simple_rnn_model

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


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch % 2) == 0:
            self.model.save("/tmp/model_{}.tf".format(epoch), save_format='tf')


def main(sequence_example_file_path):
    # read data
    ds = tf.data.TFRecordDataset([sequence_example_file_path])

    ds = ds.map(get_parse_function_shift(BASIC_EVENT_DIM))

    #ds = ds.take(10000)

    ds = ds.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([None, BASIC_EVENT_DIM], [None, BASIC_EVENT_DIM]))
    # shape is now [2             : input and label sequences
    #               128           : batch size
    #               ?             : padded sequence length per batch
    #               38]           : event dimensionality

    saver = CustomSaver()

    model = get_simple_rnn_model(BASIC_EVENT_DIM, is_Training=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(ds, epochs=50, callbacks=saver)
    model.save('/tmp/model_final.tf', save_format='tf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RNN Model")
    parser.add_argument('--sequence_example_file', dest="filename", required=True,
                        help="File containing sequence examples for training or evaluation",
                        type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()
    main(args.filename)
