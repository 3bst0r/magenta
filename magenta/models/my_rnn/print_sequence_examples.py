import itertools

import tensorflow.compat.v1 as tf
from google.protobuf.json_format import MessageToJson
import os
import argparse
import numpy as np
from magenta.common import is_valid_file
from magenta.models.my_rnn.my_rnn_generate import convert_to_note_events
from magenta.models.my_rnn.my_rnn_simple_generate import melody_seq_to_midi

tmp_dir = os.environ["TMPDIR"]

# example = tf.python_io.tf_record_iterator("/home/johannes/Documents/uni/msc/data/sequenceexamples/lmd_matched_A_lookback/training_melodies.tfrecord").__iter__().__next__()
# example = tf.train.Example.FromString(example)

# tfrecord_file = "/tmp/melody_rnn/sequence_examples/training_melodies.tfrecord"
# tfrecord_file = "/home/johannes/Documents/uni/msc/data/sequenceexamples/lmd_matched_A_lookback/training_melodies.tfrecord"
"""
tfrecord_file = "/data/johannes/data/sequence_examples/basic/training_melodies.tfrecord"
proto = tf.train.SequenceExample
# proto = note_seq.NoteSequence
for raw_bytes in tf.python_io.tf_record_iterator(tfrecord_file):
    sequence_example = proto.FromString(raw_bytes)
    sequence_example_json = MessageToJson(sequence_example)
    print(sequence_example_json)
    with open(os.path.join(tmp_dir, 'notesequence.json', 'w')) as outfile:
        outfile.write(sequence_example_json)
    break
"""


def main(sequence_example_file, midi_out_dir):
    for i, tf_record_string in enumerate(itertools.islice(tf.python_io.tf_record_iterator(sequence_example_file), 10)):
        sequence_example = tf.train.SequenceExample()
        sequence_example.ParseFromString(tf_record_string)
        one_hot_melody = np.array([input_feature.float_list.value for input_feature in
                  sequence_example.feature_lists.feature_list["inputs"].feature])
        labels = np.array([label_feature.int64_list.value[0] for label_feature in
                  sequence_example.feature_lists.feature_list["labels"].feature])
        events = np.array([np.where(one_hot == 1.)[0][0] for one_hot in one_hot_melody])
        melody_seq_to_midi(one_hot_melody, os.path.join(midi_out_dir, 'train_melody_{}.mid'.format(i)), 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RNN Model")
    parser.add_argument('--sequence_example_file', dest="filename", required=True,
                        help="File containing sequence examples for training or evaluation",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument('--midi_output_prefix', dest="midi_out_dir", required=False,
                        help="All midi training melodies will be saved in $TMP_DIR/midi_output_prefix.",
                        type=lambda x: is_valid_file(parser, os.path.join(tmp_dir, x)))
    args = parser.parse_args()
    if args.midi_out_dir:
        args.midi_out_dir = os.path.join(tmp_dir, args.midi_out_dir)
    main(args.filename, args.midi_out_dir)
