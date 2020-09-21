import tensorflow as tf
from google.protobuf.json_format import MessageToJson



#example = tf.python_io.tf_record_iterator("/home/johannes/Documents/uni/msc/data/sequenceexamples/lmd_matched_A_lookback/training_melodies.tfrecord").__iter__().__next__()
#example = tf.train.Example.FromString(example)

#tfrecord_file = "/tmp/melody_rnn/sequence_examples/training_melodies.tfrecord"
tfrecord_file = "/home/johannes/Documents/uni/msc/data/sequenceexamples/lmd_matched_A_lookback/training_melodies.tfrecord"
proto = tf.train.SequenceExample
#proto = note_seq.NoteSequence
for raw_bytes in tf.python_io.tf_record_iterator(tfrecord_file):
    sequence_example = proto.FromString(raw_bytes)
    sequence_example_json = MessageToJson(sequence_example)
    print(sequence_example_json)
    with open('/tmp/notesequence.json','w') as outfile:
        outfile.write(sequence_example_json)
    break

