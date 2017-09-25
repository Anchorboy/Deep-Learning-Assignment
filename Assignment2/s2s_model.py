# -*- coding: utf-8 -*-
import json
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf

class RNNModel:
    def __init__(self, config):
        self.config = config
        self.max_length = self.config.max_length
        self.train_iter = 0
        self.dev_iter = 0

        self.img_input_placeholder = None
        self.cap_input_placeholder = None
        self.cap_mask_placeholder = None
        self.dropout_placeholder = None

        self.tensor_list = None
        self.tensor_dict = {}

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
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(self, img_inputs_batch, cap_inputs_batch=None, mask_batch=None, dropout=1):
        feed_dict = {self.img_input_placeholder: img_inputs_batch,
                     self.dropout_placeholder:dropout}
        if cap_inputs_batch is not None:
            feed_dict[self.cap_input_placeholder] = cap_inputs_batch
        if mask_batch is not None:
            feed_dict[self.cap_mask_placeholder] = mask_batch
        return feed_dict

    def add_prediction_op(self):
        img = self.img_input_placeholder
        if self.config.mode == "train":
            cap = self.cap_input_placeholder

        dropout_rate = self.dropout_placeholder
        init = tf.truncated_normal(shape=(self.config.vocab_size, self.config.word_embed_size), stddev=1.0/ np.sqrt(self.config.vocab_size))
        word_embed = tf.get_variable('embed', initializer=init)

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
            W1 = tf.get_variable('W1', shape=(self.config.n_img_features, self.config.img_embed_size),
                                initializer=tf.contrib.layers.xavier_initializer(seed=199))
            b1 = tf.get_variable('b1', shape=(self.config.img_embed_size),
                                initializer=tf.contrib.layers.xavier_initializer(seed=198))

        with tf.variable_scope("layer2"):
            W2 = tf.get_variable('W2', shape=(self.config.hidden_size, self.config.vocab_size),
                                initializer=tf.contrib.layers.xavier_initializer(seed=123))
            b2 = tf.get_variable('b2', shape=(self.config.vocab_size),
                                 initializer=tf.contrib.layers.xavier_initializer(seed=124))

        input_shape = tf.shape(img)
        h_t1 = cell1.zero_state(input_shape[0], dtype=tf.float32)
        h_t2 = cell2.zero_state(input_shape[0], dtype=tf.float32)
        img_padding = tf.zeros([input_shape[0], self.config.img_embed_size])
        word_padding = tf.zeros([input_shape[0], self.config.word_embed_size])

        flattened_img = tf.reshape(img, shape=(-1, self.config.n_img_features))
        img_emb = tf.nn.xw_plus_b(flattened_img, W1, b1)
        img_emb = tf.nn.relu(img_emb)
        img_emb = tf.reshape(img_emb, shape=(-1, self.config.img_length, self.config.img_embed_size))

        # ---------- encoding stage ----------
        # o, h = (batch_size x hidden_size)
        self.tensor_dict['encode_h1'] = []
        self.tensor_dict['encode_h2'] = []
        self.tensor_dict['encode_o1'] = []
        for i in range(self.config.img_length):
            self.tensor_dict['encode_h1'] += [h_t1[-1]]
            self.tensor_dict['encode_h2'] += [h_t2[-1]]
            
            with tf.variable_scope("rnn1") as scope:
                if i > 0:
                    scope.reuse_variables()
                o_t1, h_t1 = cell1(img_emb[:, i, :], h_t1, scope=scope)

            with tf.variable_scope("rnn2") as scope:
                if i > 0:
                    scope.reuse_variables()
                o_t2, h_t2 = cell2(tf.concat([word_padding, h_t1[-1]], axis=1), h_t2, scope=scope)

            self.tensor_dict['encode_o1'] += [o_t1]

        # ---------- decoding stage ----------
        preds = []
        self.tensor_dict['decode_h1'] = []
        self.tensor_dict['decode_h2'] = []
        for i in range(self.config.cap_length):
            if i == 0:
                with tf.device("/cpu:0"):
                    cur_embed = tf.nn.embedding_lookup(word_embed,
                                                   tf.ones([input_shape[0]], dtype=tf.int32))
            self.tensor_dict['decode_h1'] += [h_t1[-1]]
            self.tensor_dict['decode_h2'] += [h_t2[-1]]

            with tf.variable_scope("rnn1") as scope:
                scope.reuse_variables()
                o_t1, h_t1 = cell1(img_padding, h_t1, scope=scope)

            with tf.variable_scope("rnn2") as scope:
                scope.reuse_variables()
                o_t2, h_t2 = cell2(tf.concat([cur_embed, h_t1[-1]], axis=1), h_t2, scope=scope)

            y_t = tf.nn.xw_plus_b(h_t2[-1], W2, b2)
            preds += [y_t]
            if self.config.mode == "train":
                with tf.device("/cpu:0"):
                    cur_embed = tf.nn.embedding_lookup(word_embed, cap[:, i])
            elif self.config.mode == "test":
                max_word_idx = tf.argmax(tf.nn.softmax(y_t), axis=1)
                with tf.device("/cpu:0"):
                    cur_embed = tf.nn.embedding_lookup(word_embed, max_word_idx)

        preds = tf.stack(preds, axis=1)

        self.tensor_dict = {
            'img': img_emb, 
            'encode_h1': tf.stack(self.tensor_dict['encode_h1'], axis=0), 
            'encode_h2': tf.stack(self.tensor_dict['encode_h2'], axis=0), 
            'encode_o1': tf.stack(self.tensor_dict['encode_o1'], axis=0), 
            'decode_h1': tf.stack(self.tensor_dict['decode_h1'], axis=0),
            'decode_h2': tf.stack(self.tensor_dict['decode_h2'], axis=0)
        }

        assert preds.get_shape().as_list() == [None, self.config.cap_length, self.config.vocab_size], \
            "predictions are not the right shape. Expected shape = {}, Shape = {}".format([None, self.config.cap_length, self.config.vocab_size], preds.get_shape().as_list())
        return preds

    def add_mask_op(self, preds):
        masked_logits = tf.boolean_mask(preds, self.cap_mask_placeholder)
        masked_labels = tf.boolean_mask(self.cap_input_placeholder, self.cap_mask_placeholder)
        return masked_logits, masked_labels

    def add_loss_op(self, preds):
        logits = preds
        labels = self.cap_input_placeholder
        if self.config.mode == "train":
            logits, labels = self.add_mask_op(preds)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return loss

    def add_cost_op(self, loss):
        cost = tf.reduce_mean(loss)
        return cost

    def add_training_op(self, cost):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        # compute gradients here
        grads_and_vars = optimizer.compute_gradients(cost)
        grads, variables = zip(*grads_and_vars)
        grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm=self.config.max_grad_norm)

        train_op = optimizer.apply_gradients(zip(grads, variables))
        return train_op

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.cost = self.add_cost_op(self.loss)
        if self.config.mode == "train":
            self.train_op = self.add_training_op(self.cost)
            self.sum_op = self.add_summary_op()

    def add_summary_op(self):
        tf.summary.scalar('cost', self.cost)
        merged = tf.summary.merge_all()
        return merged

# --- model operation ---

    def evaluate_dev(self, sess, data_loader, dev_writer):
        # batch = (img_slice, cap_slice, mask_slice, vid_slice)
        dev_queue = data_loader.dev_queue
        pbar = tqdm(total=dev_queue.qsize())
        total_loss = []
        while not dev_queue.empty():
            batch = dev_queue.get()
            loss, summary = self.loss_on_batch(sess, *batch[:-1])
            dev_writer.add_summary(summary, self.dev_iter)
            pbar.set_description("dev loss = {}".format(np.mean(loss)))
            pbar.update(1)
            self.dev_iter += 1
            total_loss += list(loss)
        print("")

        ave_loss = np.mean(total_loss)
        return ave_loss

    def evaluate_test(self, sess, data_loader):
        preds = []
        vids = []
        test_queue = data_loader.test_queue

        pbar = tqdm(total=test_queue.qsize(), desc="testing")
        while not test_queue.empty():
            batch = test_queue.get()
            preds_ = self.predict_on_batch(sess, *batch[:-1])
            preds += list(preds_)
            vids += list(batch[-1])
            pbar.update(1)
        print("")

        print(list(zip(vids, preds)))
        return vids, preds

    def predict_on_batch(self, sess, img_inputs_batch):
        def normalize_dict(input_dict):
            new_dict = {}
            for i, j in input_dict.items():
                new_dict[i] = j.tolist()
            return new_dict

        img_inputs_batch = np.asarray(list(img_inputs_batch), dtype=np.float32)
        feed = self.create_feed_dict(img_inputs_batch=img_inputs_batch)
        d = sess.run(self.tensor_dict, feed_dict=feed)
        print("")
        print(d['img'][0,0:3].tolist())
        print("")
        # print(d['encode_o1'][-1].tolist())
        print("")
        print("")
        # print(d['encode_h1'])
        # with open("tensor.json", "w") as f:
        #     f.write(json.dumps(normalize_dict(d)))
        # for i, j in d.items():
        #     print(i)
        #     print(j)
        #     print("")
        # print("")
        # print(d['encode_h'][-1,0])
        # print(d['encode_h'][-1,-1])
        # print(np.sqrt(np.square(d['encode_h'][-1,0] - d['encode_h'][-1,-1])))
        # p_ = sess.run(tf.nn.softmax(self.pred), feed_dict=feed)
        #print(p_)
        predictions = sess.run(tf.argmax(tf.nn.softmax(self.pred), axis=2), feed_dict=feed)
        return predictions

    def loss_on_batch(self, sess, img_inputs_batch, cap_inputs_batch, mask_batch):
        img_inputs_batch = np.asarray(list(img_inputs_batch), dtype=np.float32)
        cap_inputs_batch = np.asarray(list(cap_inputs_batch), dtype=np.int32)
        mask_batch = np.asarray(list(mask_batch), dtype=np.bool)
        feed = self.create_feed_dict(img_inputs_batch=img_inputs_batch, cap_inputs_batch=cap_inputs_batch, mask_batch=mask_batch)
        loss, _, summary = sess.run([self.loss, self.cost, self.sum_op], feed_dict=feed)
        return loss, summary

    def train_on_batch(self, sess, img_inputs_batch, cap_inputs_batch, mask_batch):
        img_inputs_batch = np.asarray(list(img_inputs_batch), dtype=np.float32)
        cap_inputs_batch = np.asarray(list(cap_inputs_batch), dtype=np.int32)
        mask_batch = np.asarray(list(mask_batch), dtype=np.bool)
        feed = self.create_feed_dict(img_inputs_batch=img_inputs_batch, cap_inputs_batch=cap_inputs_batch, mask_batch=mask_batch,
                                     dropout=self.config.dropout)
        _, loss, summary = sess.run([self.train_op, self.loss, self.sum_op], feed_dict=feed)
        return loss, summary

    def run_epoch(self, sess, data_loader, logger, writer):
        train_writer, dev_writer = writer

        # batch = (img_slice, cap_slice, mask_slice, vid_slice)
        total_loss = []
        train_queue = data_loader.train_queue
        with tqdm(total=train_queue.qsize()) as pbar:
            while not train_queue.empty():
                batch = train_queue.get()
                loss, summary = self.train_on_batch(sess, *batch[:-1])
                total_loss += list(loss)
                train_writer.add_summary(summary, self.train_iter)
                # pbar.set_description("train loss = {}".format(cost))
                pbar.update(1)
                self.train_iter += 1
        return np.mean(total_loss)

    def fit(self, sess, saver, data_loader, logger):
        best_loss = float("inf")

        train_writer = tf.summary.FileWriter(self.config.log_output + '/train',
                                             sess.graph)
        dev_writer = tf.summary.FileWriter(self.config.log_output + '/dev',
                                             sess.graph)
        for epoch in trange(self.config.n_epochs):
            loss = self.run_epoch(sess, data_loader, logger, (train_writer, dev_writer))
            if epoch % 100 == 0:
                logger.info("Epoch %d out of %d, loss = %.5f", epoch + 1, self.config.n_epochs, loss)
            # if loss < best_loss:
            #     best_loss = loss
            #     logger.info("New best score! Saving model in %s", self.config.model_output)
            if saver:
                saver.save(sess, self.config.model_output)
            # print("")
        return best_loss