# -*- coding: utf-8 -*-

import sys
import time
import logging
import argparse
import tensorflow as tf
from config import Config
from data_loader import DataLoader
from s2s_model import RNNModel

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(levelname)s:%(message)s'
logger = logging.getLogger("rnn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

def do_test(args):
    logger.info("Testing rnn model")
    args.mode = "train"
    config = Config(args)

    print(" -- loading -- ")
    data_loader = DataLoader("test")
    data_loader.load_data()
    config.vocab_size = data_loader.vocab_size
    config.max_length = data_loader.max_length

    print(" -- building -- ")
    # train_data, train_labels, train_mask
    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, data_loader, logger)

def do_train(args):
    logger.info("Training rnn model")
    args.mode = "train"
    config = Config(args)

    print(" -- loading -- ")
    data_loader = DataLoader("processed")
    data_loader.load_data()
    config.max_length = data_loader.max_length
    config.vocab_size = data_loader.vocab_size

    print(" -- building -- ")
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
            model.fit(session, saver, data_loader, logger)

def do_evaluate(args):
    logger.info("Evaluating rnn model")
    args.mode = "test"
    config = Config(args)

    print(" -- loading -- ")
    data_loader = DataLoader("test")
    data_loader.load_test_data()
    config.vocab_size = data_loader.vocab_size
    config.max_length = data_loader.max_length

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
            preds = model.evaluate_test(session, data_loader)

            # ans = ['a', 'b', 'c', 'd', 'e']
            # cols = ['id', 'answer']
            # data_frame = pd.DataFrame({"id":[i+1 for i in range(len(preds))],
            #                            "answer":[ans[i] for i in preds]})
            # data_frame.to_csv(config.output_path + "/ans.csv", index=False, sep=',', columns=cols)


def main():
    parser = argparse.ArgumentParser(description='Trains and tests an Seq2Seq model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_test)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_evaluate)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

if __name__ == "__main__":
    main()