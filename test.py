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
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("model_name", "", "Checkpoint directory from training run")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)


# initial classifier with model and vacabulary from disk 
# ==================================================
def init_classifier():
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    sess = []
    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
        # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
    return graph, sess
# get the label and score from graph through session
def label_score(graph, sess, text):
    #Argument:
    #   graph: the cnn graph
    #   sess: the session 
    #   text: the text to classify
    #Return:
    #   confidence and label index
    # Get the placeholders from the graph by name
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    # input_y = graph.get_operation_by_name("input_y").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    # Tensors we want to evaluate
    predictions = graph.get_operation_by_name("output/predictions").outputs[0]
    scores = graph.get_operation_by_name("output/scores").outputs[0]
    x_test_batch = [text]
    x_test_batch = np.array(list(vocab_processor.transform(x_test_batch)))
    #ipdb.set_trace()
    #predict = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
    #score = sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
    class_num = scores.get_shape().as_list()[-1]
    top_k = 3
    if(class_num < top_k):
        top_k = class_num
    softmax = tf.nn.softmax(scores, name="softmax")
    values, indices = tf.nn.top_k(softmax, k=top_k)
    label = []
    confidence = []
    with sess.as_default():
        label, confidence = sess.run([indices, values],{input_x: x_test_batch, dropout_keep_prob: 1.0})
    return label, confidence 
    
# Print accuracy if y_test is defined
if __name__ == "__main__":
    label = ["negative", "positive"]
    label = np.array(label)
    graph, sess = init_classifier()
    while True:
        text = raw_input("pliease input text:")
        labelIdx, confidence = label_score(graph, sess, text)
        print label[labelIdx]
        print confidence
