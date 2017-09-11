import os
import json
import pickle
import logging
import numpy as np
from config import Config

FORMAT = '%(asctime)-15s %(levelname)s:%(message)s'
logger = logging.getLogger("data_loader")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

class DataLoader:
    def __init__(self, src_dir):
        self.config = Config
        self.load_params(src_dir)
        self.load_data(src_dir)

    def load_params(self, src_dir):
        logger.info("load params")
        with open(os.path.join(src_dir, "param.plk"), "r") as f:
            vocab = pickle.load(f)
            vals = pickle.load(f)
        self.vocab = vocab
        self._vocab_size = vals['vocab_size']
        self._max_length = vals['max_length']
        logger.info("vocab size:{}, max_length:{}".format(self._vocab_size, self._max_length))

    @profile
    def load_data(self, src_dir):
        logger.info("load data")
        with open(os.path.join(src_dir, "train_label.json"), "r") as f:
            train_label = json.load(f)

        pad_vec = [0 for _ in range(self.config.img_length)]
        train_mask = {}
        for i, j in train_label.iteritems():
            mm = []
            for sent in j:
                m = [] + pad_vec
                m += [1 for _ in range(len(sent))]
                m += [0 for _ in range(self.config.max_length - len(sent) - self.config.img_length)]
                mm += [m]
            train_mask[i] = mm

        with open(os.path.join(src_dir, "test_label.json"), "r") as f:
            test_label = json.load(f)

        test_mask = {}
        for i, j in test_label.iteritems():
            mm = []
            for sent in j:
                m = [] + pad_vec
                m += [1 for _ in range(len(sent))]
                m += [0 for _ in range(self.config.max_length - len(sent) - self.config.img_length)]
                mm += [m]
            test_mask[i] = mm

def test():
    data_loader = DataLoader("preprocessed")

if __name__ == "__main__":
    test()