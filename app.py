from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
import os
from data_helpers import DataProcessor
from text_cnn import TextCNN
from tensorflow.contrib import learn
import tensorflow as tf
from tensorflow import app
import ipdb

def train():
    train_file = 'data/insure3/train.1'
    dev_file = 'data/insure3/test.1'
    log_dir = os.path.split(train_file)[0]
    #train_file = 'data/resume/train.label'
    #dev_file = 'data/resume/dev.label'
    data_processor = DataProcessor()
    data_processor.load_train_file(train_file)
    data_processor.getOriginVocab()
    data_processor.load_dev_file(dev_file)
    #data_processor.add_W('./data/GoogleNews-vectors-negative300.bin', name='resume')
    #data_processor.add_W('./data/GoogleNews-vectors-negative300.bin', name='resume')
    data_processor.add_W() # this will init W_list and W_words_map
    ipdb.set_trace()
    data_processor.saveWordsMap(output_file=os.path.join(log_dir, 'embedding.cpkl'))
    data_processor.train_text2x()
    data_processor.dev_text2x()
    graph = tf.get_default_graph()
    with graph.as_default():
        cnn = TextCNN(
            sequence_length = data_processor.maxSentenceLength,
            num_classes = len(data_processor.labels_name),
            filter_sizes = [3,4,5],
            num_filters = 128,
            VocabEmbeddings = data_processor.W_list,
            channel_num = data_processor.train_x.shape[-1],
            channel_static = [False],
            l2_reg_lambda = 0,
        )
        cnn.set_logdir(os.path.join(log_dir,'insure4'))
        cnn.Trainer(data_processor)
        #cnn.batch_cv_train(data_processor, num_epchos=5)
        #resume trainning
        #checkpoint = tf.train.latest_checkpoint('./runs/1469694510/checkpoints')
        #print 'checkpoint file', checkpoint
        #cnn.Trainer(data_processor, checkpoint)

if __name__ == '__main__':
    train()
