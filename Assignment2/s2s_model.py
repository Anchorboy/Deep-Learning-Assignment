# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class RNNModel:
    def __init__(self, config, pretrained_embeddings):
        self.config = config
        self.max_length = self.config.max_length
        self.pretrained_embeddings = pretrained_embeddings

        self.input_placeholder = None
        self.mask_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()

# --- model architecture ---

    def add_placeholders(self):
        """
        總共有幾個placeholders
        input_placeholder: (None, time_steps, )
        mask_placeholder
        labels_placeholder
        dropout_placeholder

        :return:
        """

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.max_length, self.config.n_img_features))
        self.img_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_length))
        self.cap_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_length))
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.max_length))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        feed_dict = {self.input_placeholder:inputs_batch,
                     self.mask_placeholder:mask_batch,
                     self.dropout_placeholder:dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        embed = tf.constant(self.pretrained_embeddings)
        return tf.nn.embedding_lookup(embed, self.input_placeholder)

    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        if self.config.cell == "rnn":
            cell = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size)
        elif self.config.cell == "gru":
            cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        elif self.config.cell == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
        else:
            raise ValueError("Unsupported cell type: " + self.config.cell)

        if self.config.mode == "train":
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate)

        with tf.variable_scope("layer1"):
            U = tf.get_variable('U', shape=(self.config.hidden_size, self.config.vocab_size),
                                initializer=tf.contrib.layers.xavier_initializer(seed=123))
            b2 = tf.get_variable('b2', shape=(self.config.vocab_size),
                                 initializer=tf.contrib.layers.xavier_initializer(seed=124))

        input_shape = tf.shape(x)
        with tf.variable_scope("RNN"):
            outputs, states = tf.nn.dynamic_rnn(cell, x, time_major=False, dtype=tf.float32)
        outputs = tf.reshape(outputs, shape=[-1, self.config.hidden_size])

        preds = tf.matmul(outputs, U) + b2
        preds = tf.reshape(preds, shape=[-1, self.max_length, self.config.vocab_size])
        assert preds.get_shape().as_list() == [None, self.max_length, self.config.vocab_size], \
            "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.vocab_size], preds.get_shape().as_list())
        return preds

    def add_mask_op(self, preds):
        masked_logits = tf.boolean_mask(preds, self.mask_placeholder)
        masked_labels = tf.boolean_mask(self.labels_placeholder, self.mask_placeholder)
        return masked_logits, masked_labels

    def add_loss_op(self, preds):
        logits = preds
        labels = self.labels_placeholder
        if self.config.mode == "train":
            logits, labels = self.add_mask_op(preds=preds)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return loss

    def add_cost_op(self, loss):
        cost = tf.reduce_mean(loss)
        return cost

    def add_training_op(self, cost):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(cost)
        return train_op

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.cost = self.add_cost_op(self.loss)
        self.train_op = self.add_training_op(self.cost)

        self.evals = self.add_eval_test_op(self.loss)
        self.sum_op = self.add_summary_op()

    def add_eval_test_op(self, loss):
        cost_list = tf.reduce_mean(loss, axis=1)
        cost_list = tf.reshape(cost_list, shape=[-1, 5])
        evals = tf.argmin(cost_list, axis=1)
        return evals

    def evaluate_test(self, sess, data_loader):
        preds = []
        n_test_batch = data_loader.n_test_batch(self.config.batch_size)
        pbar = tqdm(total=n_test_batch, desc='test eval')
        for i, batch in enumerate(data_loader.test_batch(self.config.batch_size)):
            preds_ = self.evaluate_on_batch(sess, *batch)
            preds += list(preds_)
            pbar.update(1)
        return preds

    def evaluate_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        # original shape = [?, num_candidates, max_length]
        # reshape to -> [?, max_length]
        inputs_batch = np.reshape(inputs_batch, [-1, self.config.max_length])
        labels_batch = np.reshape(labels_batch, [-1, self.config.max_length])
        mask_batch = np.reshape(mask_batch, [-1, self.config.max_length])
        feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                     labels_batch=labels_batch,
                                     mask_batch=mask_batch)

        preds_ = sess.run(self.evals, feed_dict=feed)
        return preds_

    def add_summary_op(self):
        tf.summary.scalar('cost', self.cost)
        merged = tf.summary.merge_all()
        return merged

# --- model operation ---

    def output(self, sess, n_batch, inputs_batch, desc, data):
        preds = []
        pbar = tqdm(total=n_batch, desc=desc)
        for i, batch in enumerate(inputs_batch):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            pbar.update(1)

        preds = np.asarray(preds, dtype=np.int32)
        _, labels, mask = data
        masked_preds = preds[mask]
        masked_labels = labels[mask]
        acc = np.mean(masked_preds == masked_labels)
        return acc

    def evaluate(self, sess, data_loader):
        # --- train ---
        n_train_batch = data_loader.n_train_batch(self.config.batch_size)
        train_batch = data_loader.train_batch(self.config.batch_size)
        train_data = data_loader.train_data
        train_acc = self.output(sess, n_train_batch, train_batch, 'train eval', train_data)

        # --- dev ---
        n_dev_batch = data_loader.n_dev_batch(self.config.batch_size)
        dev_batch = data_loader.dev_batch(self.config.batch_size)
        dev_data = data_loader.dev_data
        dev_acc = self.output(sess, n_dev_batch, dev_batch, 'dev eval', dev_data)

        return train_acc, dev_acc

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=self.config.dropout)
        _, cost, summary = sess.run([self.train_op, self.cost, self.sum_op], feed_dict=feed)
        return cost, summary

    def run_epoch(self, sess, data_loader, logger, writer):
        train_writer, dev_writer = writer

        n_train_batch = data_loader.n_train_batch(self.config.batch_size)
        pbar = tqdm(total=n_train_batch)
        for i, batch in enumerate(data_loader.train_batch(self.config.batch_size)):
            cost, summary = self.train_on_batch(sess, *batch)
            train_writer.add_summary(summary, i)
            pbar.set_description("train loss = {}".format(cost))
            pbar.update(1)
        print("")

        logger.info("Evaluating on training data")
        acc = self.evaluate(sess, data_loader)
        logger.info("Accuracy on train/dev: %.4f/%.4f", *acc)

        return acc[1]

    def fit(self, sess, saver, data_loader, logger):
        best_score = 0.

        train_writer = tf.summary.FileWriter(self.config.log_output + '/train',
                                             sess.graph)
        dev_writer = tf.summary.FileWriter(self.config.log_output + '/dev',
                                             sess.graph)
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, data_loader, logger, (train_writer, dev_writer))
            if score > best_score:
                best_score = score
                logger.info("New best score! Saving model in %s", self.config.model_output)
            if saver:
                saver.save(sess, self.config.model_output)
            print("")
        return best_score