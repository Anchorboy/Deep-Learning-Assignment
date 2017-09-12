import os
import json
import pickle
import logging
import numpy as np
from config import Config
from os.path import join as pjoin
import queue

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

    def process_dict(self, dir, input_dict):
        vid_list = input_dict['id']
        cap_list = []
        np_list = map(lambda x: np.load(pjoin(dir, x + ".npy")), vid_list)

        mask_list = []
        for sent in input_dict['caption']:
            m = [1 for _ in range(len(sent))]
            pad = [0 for _ in range(self.config.cap_length - len(sent))]
            m += pad
            assert len(m) == self.config.cap_length
            mask_list += [m]
            cap_list += [sent + pad]

        input_dict['caption'] = np.asarray(cap_list, dtype=np.int32)
        input_dict['mask'] = np.asarray(mask_list, dtype=np.bool)
        input_dict['img'] = np.asarray(np_list, dtype=np.float32)

        return input_dict

    def load_data(self, src_dir):
        logger.info("load data")
        # ----- training data -----
        with open(pjoin(src_dir, "train_label.json"), "r") as f:
            train_data = json.load(f)
        train_dir = pjoin("MLDS_hw2_data", "training_data", "feat")
        train_data = self.process_dict(train_dir, train_data)

        # ----- developing data -----
        with open(pjoin(src_dir, "dev_label.json"), "r") as f:
            dev_data = json.load(f)
        dev_data = self.process_dict(train_dir, dev_data)

        # ----- testing data -----
        with open(pjoin(src_dir, "test_label.json"), "r") as f:
            test_data = json.load(f)
        test_dir = pjoin("MLDS_hw2_data", "testing_data", "feat")
        test_data = self.process_dict(test_dir, test_data)

        self._train_data = train_data
        self._dev_data = dev_data
        self._test_data = test_data

    def data_queue(self, data):
        """

        :param data:
        :return: (img_slice, cap_slice, mask_slice, vid_slice)
        """
        q = queue.Queue()

        vid_list = data['id']
        img_list = data['img']
        cap_list = data['caption']
        mask_list = data['mask']

        assert cap_list.shape[0] == mask_list.shape[0] == img_list.shape[0]
        n_samples = cap_list.shape[0]
        idx = np.arange(n_samples)
        np.random.shuffle(idx)

        start = 0
        end = n_samples
        while start < end:
            cur_end = start + self.config.batch_size
            indices = idx[start:cur_end]
            img_slice = map(lambda x: img_list[x], indices)
            vid_slice = map(lambda x: vid_list[x], indices)
            cap_slice = map(lambda x: cap_list[x], indices)
            mask_slice = map(lambda x: mask_list[x], indices)
            q.put((img_slice, cap_slice, mask_slice, vid_slice))

            start = cur_end
        return q

    @property
    def train_queue(self):
        return self.data_queue(self.train_data)

    @property
    def dev_queue(self):
        return self.data_queue(self.dev_data)

    @property
    def test_queue(self):
        return self.data_queue(self.test_data)

    @property
    def train_data(self):
        return self._train_data

    @property
    def dev_data(self):
        return self._dev_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_length(self):
        return self._max_length

def test():
    data_loader = DataLoader("test")
    train_queue = data_loader.train_queue
    print(train_queue)
    # data_loader.data_queue(data_loader.train_data)

if __name__ == "__main__":
    test()