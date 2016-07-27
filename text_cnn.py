import tensorflow as tf
import numpy as np
import cPickle
import ipdb

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      filter_sizes, num_filters, VocabEmbeddings, channel_num, l2_reg_lambda=0.0):
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
            for idx, vocab in enumerate(VocabEmbeddings):
                if vocab['static']:
                    print 'name:{0}, static , embedding_size: {1}'.format(vocab['name'],vocab['embedding'].shape[-1])
                    W = tf.constant(vocab['embedding'], name=vocab['name'])
                else:
                    print 'name:{0}, non-static , embedding_size: {1}'.format(vocab['name'],vocab['embedding'].shape[-1])
                    W = tf.Variable(vocab['embedding'], name=vocab['name'])
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

    def Trainer(W_list, x_list, y_list):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=True,
              log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                self = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=FLAGS.class_num,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    VocabEmbeddings=VocabEmbedModels,
                    channel_num=len(VocabEmbedModels),
                    )

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

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
                #timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H_%M_%S")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.scalar_summary("loss", self.loss)
                acc_summary = tf.scalar_summary("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)

                # Write vocabulary
                # vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables
                sess.run(tf.initialize_all_variables())
                
    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, self.loss, self.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)
    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, self.loss, self.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    def batch_train():
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
