# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # train = (55000, 784) = (n_samples, n_features)
    n_features = 784
    n_classes = 10
    hidden_size = 200
    n_epochs = 5000
    # Create the model

    # create a hidden layer with relu & output layer with softmax

    with tf.variable_scope("mnist_softmax"):
        x = tf.placeholder(dtype=tf.float32, shape=(None, n_features))
        y = tf.placeholder(dtype=tf.int32, shape=(None, n_classes))
        W1 = tf.Variable(tf.random_uniform(shape=(n_features, hidden_size), minval=-1, maxval=1))
        b1 = tf.Variable(tf.zeros(shape=(hidden_size,)))

        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        W2 = tf.Variable(tf.random_uniform(shape=(hidden_size, n_classes), minval=-1, maxval=1))
        b2 = tf.Variable(tf.zeros(shape=(n_classes,)))
        y_ = tf.matmul(h1, W2) + b2

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
        train_ops = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        # init shoud be right before Session start!!!
        init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        ave_loss = 0.0
        for i in range(n_epochs):
            batch_x, batch_y = mnist.train.next_batch(100)
            feed_dict = {x: batch_x, y: batch_y}
            _, loss_val = sess.run([train_ops, loss], feed_dict=feed_dict)
            if i % 100 == 0:
                print("loss at step =", i, "loss_val =", loss_val)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y: mnist.test.labels}))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    #                     help='Directory for storing input data')
    # FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main)
