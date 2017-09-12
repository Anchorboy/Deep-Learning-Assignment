# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class RNNModel:
    def __init__(self, config):
        self.config = config
        self.max_length = self.config.max_length
        self.iter = 0

        self.img_input_placeholder = None
        self.cap_input_placeholder = None
        self.cap_mask_placeholder = None
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

        self.img_input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.config.img_length, self.config.n_img_features))
        self.cap_input_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.config.cap_length))
        self.cap_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.config.cap_length))
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.config.cap_length))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(self, img_inputs_batch, cap_inputs_batch, mask_batch=None, labels_batch=None, dropout=1):
        feed_dict = {self.img_input_placeholder: img_inputs_batch,
                     self.cap_input_placeholder: cap_inputs_batch,
                     self.dropout_placeholder:dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if mask_batch is not None:
            feed_dict[self.cap_mask_placeholder] = mask_batch
        return feed_dict

    def add_word_embedding(self):
        embed = tf.get_variable('embed', shape=(self.config.vocab_size, self.config.embed_size),
                                initializer=tf.truncated_normal(shape=(self.config.vocab_size, self.config.embed_size), stddev=1.0/ np.sqrt(self.config.vocab_size)))
        return embed

    def add_prediction_op(self):
        img = self.img_input_placeholder
        cap = self.cap_input_placeholder
        word_embed = self.embed
        dropout_rate = self.dropout_placeholder

        if self.config.cell == "rnn":
            cell1 = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size)
            cell2 = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size)
        elif self.config.cell == "gru":
            cell1 = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            cell2 = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        elif self.config.cell == "lstm":
            cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
        else:
            raise ValueError("Unsupported cell type: " + self.config.cell)

        if self.config.mode == "train":
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=dropout_rate)
            cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=dropout_rate)

        with tf.variable_scope("layer1"):
            W1 = tf.get_variable('W1', shape=(self.config.hidden_size, self.config.vocab_size),
                                initializer=tf.contrib.layers.xavier_initializer(seed=123))
            b1 = tf.get_variable('b1', shape=(self.config.vocab_size),
                                 initializer=tf.contrib.layers.xavier_initializer(seed=124))

        input_shape = tf.shape(img)
        h1 = cell1.zero_state(input_shape, dtype=tf.float32)
        h2 = cell2.zero_state(input_shape, dtype=tf.float32)
        padding = tf.zeros([input_shape[0], self.config.hidden_size])

        # ---------- encoding stage ----------
        for i in range(self.config.img_length):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("rnn1") as scope:
                o_t1, h_t1 = cell1(img[:, i, :], h1, scope=scope)

            with tf.variable_scope("rnn2") as scope:
                o_t2, h_t2 = cell2(tf.concat([padding, h_t1], axis=1), h2, scope=scope)

        # ---------- decoding stage ----------
        preds = []
        for i in range(self.config.cap_length):
            with tf.device("/cpu:0"):
                cur_embed = tf.nn.embedding_lookup(word_embed, cap[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("rnn1") as scope:
                o_t1, h_t1 = cell1(padding, h1, scope=scope)

            with tf.variable_scope("rnn2") as scope:
                o_t2, h_t2 = cell2(tf.concat([cur_embed, h_t1], axis=1), h2, scope=scope)

            y_t = tf.nn.xw_plus_b(o_t2, W1, b1)
            preds += [y_t]

        preds = tf.stack(preds, axis=1)

        assert preds.get_shape().as_list() == [None, self.config.cap_length, self.config.vocab_size], \
            "predictions are not the right shape. Expected shape = {}, Shape = {}".format([None, self.config.cap_length, self.config.vocab_size], preds.get_shape().as_list())
        return preds

    def add_mask_op(self, preds):
        masked_logits = tf.boolean_mask(preds, self.cap_mask_placeholder)
        masked_labels = tf.boolean_mask(self.labels_placeholder, self.cap_mask_placeholder)
        return masked_logits, masked_labels

    def add_loss_op(self, preds):
        logits = preds
        labels = self.labels_placeholder
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
        self.embed = self.add_word_embedding()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.cost = self.add_cost_op(self.loss)
        self.train_op = self.add_training_op(self.cost)

        self.sum_op = self.add_summary_op()

    def add_summary_op(self):
        tf.summary.scalar('cost', self.cost)
        merged = tf.summary.merge_all()
        return merged

# --- model operation ---

    """
    def output(self, sess, inputs_queue, desc, data):
        vids = []
        preds = []
        pbar = tqdm(total=inputs_queue.qsize(), desc=desc)
        while not inputs_queue.empty():
            # Ignore predict
            # batch = (img_slice, cap_slice, mask_slice, vid_slice)
            batch = inputs_queue.get()
            preds_ = self.predict_on_batch(sess, *batch[:-1])
            preds += list(preds_)
            vids += batch[-1]
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
        

    def predict_on_batch(self, sess, img_inputs_batch, cap_inputs_batch, mask_batch):
        feed = self.create_feed_dict(img_inputs_batch=img_inputs_batch, cap_inputs_batch=cap_inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(tf.nn.softmax(self.pred), axis=2), feed_dict=feed)
        return predictions
        """

    def train_on_batch(self, sess, img_inputs_batch, cap_inputs_batch, cap_labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=self.config.dropout)
        _, cost, summary = sess.run([self.train_op, self.cost, self.sum_op], feed_dict=feed)
        return cost, summary

    def run_epoch(self, sess, data_loader, logger, writer):
        train_writer, dev_writer = writer

        # batch = (img_slice, cap_slice, mask_slice, vid_slice)
        train_queue = data_loader.train_queue
        pbar = tqdm(total=train_queue.qsize())
        while not train_queue.empty():
            batch = train_queue.get()
            cost, summary = self.train_on_batch(sess, *batch[:-1])
            train_writer.add_summary(summary, self.iter)
            pbar.set_description("train loss = {}".format(cost))
            pbar.update(1)
            self.iter += 1
        print("")

        # logger.info("Evaluating on training data")
        # acc = self.evaluate(sess, data_loader)
        # logger.info("Accuracy on train/dev: %.4f/%.4f", *acc)

        # return acc[1]

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