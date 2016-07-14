#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

import ipdb

tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("model_file", "", "model file to restore the graph")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



def init_classifier():
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if FLAGS.model_file:
        checkpoint_file = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_file)
        print "load the graph from file{}......".format(checkpoint_file)
    graph = tf.Graph()
    #with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    #with sess.as_default():
    # Load the saved meta graph and restore variables
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
    ipdb.set_trace()
    return graph, sess

def label_score(graph, sess, input_text):
    # Get the placeholders from the graph by name
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    # input_y = graph.get_operation_by_name("input_y").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    # Tensors we want to evaluate
    predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    input_text = [input_text]
    x_test = np.array(list(vocab_processor.transform(input_text)))
    ipdb.set_trace()
    with sess.as_default():
        label_score = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})

    values, indices = tf.nn.top_k(label_score, k = 5, sorted=True)
    return values, indices

if __name__ == "__main__":
    graph, sess = init_classifier()
    ipdb.set_trace()
    values, indices = label_score(graph, sess, "it's greate! beautiful flower")
    print values, indices

