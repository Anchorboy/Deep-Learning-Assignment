import tensorflow as tf

with tf.variable_scope("rnn"):
    with tf.name_scope("encoder"):
        c1 = tf.nn.rnn_cell.BasicLSTMCell(256, reuse=True)
    #     a = tf.zeros((10,))
    #
    # with tf.name_scope("decoder"):
    #     b = tf.zeros((10,))

def lstm_cell():
    cell = tf.contrib.rnn.NASCell(256, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)

def test():
    x = tf.get_variable('x', shape=(64, 100, 100))
    with tf.variable_scope("RNN"):
        cell = tf.nn.rnn_cell.BasicLSTMCell(256)

        with tf.variable_scope("encoder"):
            outputs, states = tf.nn.dynamic_rnn(cell, x, time_major=False, dtype=tf.float32)

        # print outputs, states

        with tf.variable_scope("decoder"):
            outputs, states = tf.nn.dynamic_rnn(cell, x, time_major=False, dtype=tf.float32)

        # print outputs, states

if __name__ == "__main__":
    test()