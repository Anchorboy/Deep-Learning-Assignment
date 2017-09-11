import sys
import logging
import argparse
from datetime import *
import time
import numpy as np
from tqdm import tqdm
from tqdm import trange
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.examples.tutorials.mnist import input_data

logger = logging.getLogger("rnn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    max_length = 28
    n_input = 28
    n_classes = 10
    dropout = 0.667
    embed_size = 50
    hidden_size = 128
    img_size = 28 * 28
    batch_size = 100
    n_epochs = 10
    lr = 0.001
    max_grad_norm = 5.

    def __init__(self, args):
        self.cell = args.cell
        self.mode = args.mode

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"

class RNNModel:
    def __init__(self, config):
        self.config = config

        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.config.max_length, self.config.n_input))
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.config.n_classes))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1.):
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        x = self.input_placeholder
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
            U = tf.get_variable('U', shape=(self.config.hidden_size, self.config.n_classes), initializer=tf.contrib.layers.xavier_initializer(seed=123))
            b2 = tf.get_variable('b2', shape=(self.config.n_classes), initializer=tf.contrib.layers.xavier_initializer(seed=124))

        x_transpose = tf.transpose(x, [1, 0, 2])

        with tf.variable_scope("RNN"):
            outputs, states = tf.nn.dynamic_rnn(cell, inputs=x_transpose, time_major=True, dtype=tf.float32)

        # outputs = tf.transpose(outputs, [1, 0, 2])
        preds = tf.matmul(outputs[-1], U) + b2
        # preds = tf.matmul(states[-1], U) + b2
        assert preds.get_shape().as_list() == [None, self.config.n_classes], \
            "predictions are not of the right shape. Expected {}, got {}".format(
                [None, self.config.n_classes], preds.get_shape().as_list())
        return preds

    def add_loss_op(self, preds):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=self.labels_placeholder))
        return loss

    def add_sum_op(self):
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all()
        return merged

    def evaluate_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch)
        preds = tf.argmax(tf.nn.softmax(self.pred), axis=1)
        preds_ = sess.run(preds, feed_dict=feed)
        return preds_

    def evaluate(self, sess, dataset):
        n_batch = dataset.num_examples // self.config.batch_size + 1 \
            if dataset.num_examples % self.config.batch_size != 0 else dataset.num_examples // self.config.batch_size

        preds = []
        pbar = trange(n_batch)
        for i in pbar:
            batch_x, batch_y = dataset.next_batch(self.config.batch_size)
            batch_x = np.reshape(batch_x, [-1, self.config.max_length, self.config.n_input])
            preds_ = self.evaluate_on_batch(sess, batch_x)
            preds += list(preds_)
            pbar.update(1)
        print("")

        preds = np.asarray(preds, dtype=np.int32)
        labels = np.argmax(dataset.labels, axis=1)
        correct_prediction = preds == labels
        acc = np.mean(correct_prediction)
        return acc

    def add_training_op_without_clip(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)

        grads_and_vars = optimizer.compute_gradients(loss=loss)
        grads, variables = zip(*grads_and_vars)

        grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm=self.config.max_grad_norm)

        train_op = optimizer.apply_gradients(zip(grads, variables))
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss_val, summary = sess.run([self.train_op, self.loss, self.sum_op], feed_dict=feed)
        return summary, loss_val

    def run_epoch(self, sess, mnist, writer):

        n_train_batch = mnist.train.num_examples // self.config.batch_size + 1 \
                if mnist.train.num_examples % self.config.batch_size != 0 else mnist.train.num_examples // self.config.batch_size

        pbar = trange(n_train_batch)
        for i in pbar:
            batch_x, batch_y = mnist.train.next_batch(self.config.batch_size)
            batch_x = np.reshape(batch_x, [-1, self.config.max_length, self.config.n_input])
            summary, loss = self.train_on_batch(sess, inputs_batch=batch_x, labels_batch=batch_y)
            writer.add_summary(summary, i)
            pbar.set_description("train loss = {}".format(loss))
            pbar.update(1)
        print("")

        logger.info("Evaluating on training data")
        train_acc = self.evaluate(sess, mnist.train)
        logger.info("Accuracy: %.4f", train_acc)

        logger.info("Evaluating on validation data")
        valid_acc = self.evaluate(sess, mnist.validation)
        logger.info("Accuracy: %.4f", valid_acc)
        return valid_acc

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        # self.train_op = self.add_training_op_without_clip(self.loss)

        self.sum_op = self.add_sum_op()

    def fit(self, sess, saver, mnist, logger):
        best_score = 0.

        train_writer = tf.summary.FileWriter(self.config.log_output + '/train',
                                             sess.graph)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, mnist, train_writer)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")

        return best_score

def do_train(args):
    logger.info("Training rnn model")
    args.mode = "train"
    config = Config(args)

    print " -- loading -- "
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print " -- building -- "
    # train_data, train_labels, train_mask
    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = RNNModel(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, mnist, logger)

def do_evaluate(args):
    logger.info("Training rnn model")
    args.mode = "test"
    config = Config(args)

    print " -- loading -- "
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print " -- building -- "
    # train_data, train_labels, train_mask
    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = RNNModel(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            logger.info("Evaluating test set")
            acc = model.evaluate(session, dataset=mnist.test)
            logger.info("Accuracy on test: %.4f", acc)

def main():
    parser = argparse.ArgumentParser(description='Trains and tests an Seq2Seq model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn",
                                help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn",
                                help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_evaluate)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

if __name__ == "__main__":
    main()