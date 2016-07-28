import tensorflow as tf
import numpy as np
import time
import datetime
import cPickle
import os
import ipdb


#tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
#tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
#FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      filter_sizes, num_filters, VocabEmbeddings, channel_num, channel_static, l2_reg_lambda=0.0):
        """
            sequence_length: the length of every sentence
            VocabEmbeddings: a list contains the different embeddings for multi channel
                in which element is a dict {name:'w2v', embedding:w2v_matrix, static:True}
                and the num equals to channel_num
            the input_x's channel dimension is corronspandence to each embedding in VocabEmbeddings
        """
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        Embeddings = [] 
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            for idx, W in enumerate(VocabEmbeddings):
                if channel_static[idx]:
                    print 'static , embedding_size: {}'.format(W.shape[-1])
                    W = tf.constant(W, name='static')
                else:
                    print 'non-static , embedding_size: {}'.format(W.shape[-1])
                    W = tf.Variable(W, name='non_static')
                embedded_chars = tf.nn.embedding_lookup(W, self.input_x[:,:,idx])
                Embeddings.append(tf.expand_dims(embedded_chars, -1))

        # Create a convolution + maxpool layer for each filter size
        # below take the all the W as same size, so just concat them
        embedding = tf.concat(concat_dim=3, values=Embeddings, name="concat_embeddings")
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                embedding_size = embedding.get_shape().as_list()[-2]
                filter_shape = [filter_size, embedding_size, channel_num, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedding,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # add l2_norm based on original paper
            W = tf.nn.l2_normalize(W,dim=0) * 3.0
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def Trainer(self, data_processor, resume_train_checkpoint = None):
    # resume_train is checkpoint file path
        graph = tf.get_default_graph()
        #with tf.Graph().as_default():
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=True,
              log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Define Training procedure
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(loss=self.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
                
                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.merge_summary(grad_summaries)
                
                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.scalar_summary("loss", self.loss)
                acc_summary = tf.scalar_summary("accuracy", self.accuracy)

                # Train Summaries
                self.train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                self.train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

                # Dev summaries
                self.dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                self.dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                self.saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)
                if resume_train_checkpoint is None:
                    # Initialize all variables
                    sess.run(tf.initialize_all_variables())
                    self.batch_train(sess, data_processor, batch_size=64, num_epochs=100)
                else:
                    self.saver.restore(sess, resume_train_checkpoint)
                    print("Model restored.")
                    self.batch_train(sess, data_processor, batch_size=64, num_epochs=100)
    def train_step(self, x_batch, y_batch, sess):
        """
        A single training step
        """
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob: 0.5 #FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)
    def dev_step(self, x_batch, y_batch, sess, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [self.global_step, self.dev_summary_op, self.loss, self.accuracy],
            feed_dict)
        #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
        return loss, accuracy

    def batch_train(self, sess, data_processor, batch_size=64, num_epochs=100):
        # Generate batches
        train_batches = data_processor.batch_iter(data_processor.train_x, data_processor.train_y,
            batch_size, num_epochs)
        dev_num = data_processor.dev_x.shape[0]
        # Training loop. For each batch...
        for train_batch in train_batches:
            train_x_batch, train_y_batch = zip(*train_batch)
            self.train_step(train_x_batch, train_y_batch, sess)
            current_step = tf.train.global_step(sess, self.global_step)
            #if current_step % FLAGS.evaluate_every == 0:
            if current_step % 100 == 0:
                print("\nEvaluation:")
                if dev_num < 30000:
                    loss, accuracy = self.dev_step(
                        data_processor.dev_x, data_processor.dev_y, 
                        sess, writer=self.dev_summary_writer)
                else:
                    dev_batches = data_processor.batch_iter(
                        data_processor.dev_x, data_processor.dev_y,
                        batch_size=10000, num_epochs=1, shuffle=True)
                    loss, accuracy = [], [];
                    for dev_batch in dev_batches:
                        dev_x_batch, dev_y_batch = zip(*dev_batch)
                        loss_batch, accuracy_batch = self.dev_step(
                            dev_x_batch, dev_y_batch, sess, writer=self.dev_summary_writer)
                        loss.append(loss_batch)
                        accuracy.append(accuracy_batch)
                    loss = float(sum(loss)) / len(loss)
                    accuracy = float(sum(accuracy)) / len(accuracy)
                step = self.global_step.eval()
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                print("")
            #if current_step % FLAGS.checkpoint_every == 0:
            if current_step % 100 == 0:
                path = self.saver.save(sess, self.checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
