# %%
import os
import argparse
# import pydevd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from magenta.models.my_rnn.my_simple_rnn_model import BASIC_EVENT_DIM
from magenta.models.my_rnn.my_simple_rnn_model import get_simple_rnn_model
from magenta.common import is_valid_file
from magenta.models.my_rnn.my_rnn_generate import melody_seq_to_midi
import uuid

BATCH_SIZE = 128
NUM_THREADS = 7
tmp_dir = os.environ["TMPDIR"]

# tf.disable_eager_execution()
print("executing eagerly:")
print(tf.executing_eagerly())
# tf.config.experimental_run_functions_eagerly(True)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)


def get_parse_function_shift(event_dim, label_shape=None):
    sequence_features = {
        'inputs': tf.io.FixedLenSequenceFeature(shape=[event_dim],
                                                dtype=tf.float32),
        'labels': tf.io.FixedLenSequenceFeature(shape=label_shape or [],
                                                dtype=tf.int64)}

    def shift_melody(example):
        # one example is one melody
        _, sequence = tf.parse_single_sequence_example(serialized=example,
                                                       sequence_features=sequence_features)
        # return melody from first step as input and melody shifted by one step to the left as label
        return sequence['inputs'][:-1], sequence['inputs'][1:]

    return shift_melody


class CustomSaver(keras.callbacks.Callback):

    def __init__(self, model_prefix):
        self.model_prefix = model_prefix

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # if (epoch % 2) == 0:
        self.model.save(os.path.join(self.model_prefix, "model_{}.tf".format(epoch)), save_format='tf')


def main(sequence_example_file_path, model_prefix):
    print(tf.__version__)

    # read data
    ds = tf.data.TFRecordDataset(sequence_example_file_path)

    #ds = ds.map(get_parse_function_shift(BASIC_EVENT_DIM))

    #ds = ds.shuffle(buffer_size=2048)

    ds = ds.take(int(1E5))

    ds = ds.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([None, BASIC_EVENT_DIM], [None, BASIC_EVENT_DIM]))
    # shape is now [2             : input and label sequences
    #               128           : batch size
    #               ?             : padded sequence length per batch
    #               38]           : event dimensionality

    saver = CustomSaver(model_prefix)

    model = get_simple_rnn_model(BASIC_EVENT_DIM, is_Training=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(ds, epochs=25, callbacks=[saver])
    model.save(os.path.join(model_prefix, 'model_final.tf'), save_format='tf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RNN Model")
    parser.add_argument('--sequence_example_file', dest="filename", required=True,
                        help="File containing sequence examples for training or evaluation",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument('--prefix', dest="prefix", required=True,
                        help="All model iterations will be saved in $TMP_DIR/prefix/")
    args = parser.parse_args()
    main(sequence_example_file_path=args.filename, model_prefix=os.path.join(tmp_dir, args.prefix))
