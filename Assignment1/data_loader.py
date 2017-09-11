import os
import json
import pickle
import numpy as np

class DataLoader:
    def __init__(self, datadir):
        self._datadir = datadir
        self.load_params()

    def load_params(self):
        print "load param"
        with open(os.path.join(self._datadir, "param.plk"), "r") as f:
            vocab = pickle.load(f)
            vals = pickle.load(f)
        self.vocab = vocab
        self._vocab_size = vals['vocab_size']
        self._max_length = vals['max_length']

    def preprocess(self, input_dict):
        # input_dict['data'] = np.asarray(input_dict['data'], dtype=np.int32)
        # input_dict['mask'] = input_dict['data'] != 0
        data = np.asarray(input_dict['data'], dtype=np.int32)
        mask = data != 0
        labels = np.zeros_like(input_dict['data'])
        labels[:, :-1] += data[:, 1:]
        input_dict['data'] = data
        input_dict['mask'] = mask
        input_dict['labels'] = labels
        return input_dict

    def load_and_preprocess(self):
        print "load data"
        with open(os.path.join(self._datadir, "train.json"), "r") as f:
            train = json.loads(f.read())

        with open(os.path.join(self._datadir, "dev.json"), "r") as f:
            dev = json.loads(f.read())

        print "preprocess"
        self.train = self.preprocess(train)
        self.dev = self.preprocess(dev)

    def load_and_preprocess_test(self):
        print "load test data"
        with open(os.path.join(self._datadir, "test.json"), "r") as f:
            test = json.loads(f.read())

        print "preprocess"
        self.test = self.preprocess(test)

    def load_embedding(self):
        print "load embedding"
        tokens = {}
        loaded_embed_mat = []
        with open(os.path.join(self._datadir, "glove.6B.100d.txt"), "r") as f:
            idx = 0
            for line in f:
                line = line.strip().split()
                wd = line[0]
                vec = line[1:]
                tokens[wd] = idx
                loaded_embed_mat += [vec]
                idx += 1
        loaded_embed_mat = np.asarray(loaded_embed_mat, dtype=np.float32)

        embedding_size = len(loaded_embed_mat[0])
        embed_mat = np.zeros((self.vocab_size, embedding_size), dtype=np.float32)

        for wd, idx in self.vocab.iteritems():
            if wd in tokens:
                embed_mat[idx] += loaded_embed_mat[tokens[wd]]

        return embedding_size, embed_mat

    def get_batch(self, batch_size, input_dict):
        n_samples = input_dict['n_samples']
        n_batch = self.n_batch(n_samples, batch_size)
        # n_batch = n_samples // batch_size + 1 if n_samples % batch_size == 0 else n_samples // batch_size

        for i in range(n_batch):
            yield [input_dict['data'][i * batch_size: (i + 1) * batch_size],
                   input_dict['labels'][i * batch_size: (i + 1) * batch_size],
                   input_dict['mask'][i * batch_size: (i + 1) * batch_size]]

    def get_test_batch(self, batch_size, input_dict):
        n_cands = 5
        # batch_size x n_cands x max_length
        n_samples = input_dict['n_samples'] // n_cands
        n_batch = self.n_batch(n_samples, batch_size)

        data = np.reshape(input_dict['data'], (-1, n_cands, self.max_length))
        labels = np.reshape(input_dict['labels'], (-1, n_cands, self.max_length))
        mask = np.reshape(input_dict['mask'], (-1, n_cands, self.max_length))
        for i in range(n_batch):
            yield [data[i * batch_size: (i+1) * batch_size],
                   labels[i * batch_size: (i+1) * batch_size],
                   mask[i * batch_size: (i+1) * batch_size]]


    def n_batch(self, n_samples, batch_size):
        return n_samples // batch_size + 1 if n_samples % batch_size != 0 else n_samples // batch_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_length(self):
        return self._max_length

    def n_train_batch(self, batch_size=64):
        return self.n_batch(self.train['n_samples'], batch_size)

    def n_dev_batch(self, batch_size=64):
        return self.n_batch(self.dev['n_samples'], batch_size)

    def n_test_batch(self, batch_size=64):
        n_cands = 5
        return self.n_batch(self.test['n_samples']//n_cands, batch_size)

    @property
    def train_data(self):
        return [self.train['data'], self.train['labels'], self.train['mask']]

    @property
    def dev_data(self):
        return [self.dev['data'], self.dev['labels'], self.dev['mask']]

    @property
    def test_data(self):
        return [self.test['data'], self.test['labels'], self.test['mask']]

    def train_batch(self, batch_size=64):
        return self.get_batch(batch_size, self.train)

    def dev_batch(self, batch_size=64):
        return self.get_batch(batch_size, self.dev)

    def test_batch(self, batch_size=64):
        return self.get_test_batch(batch_size, self.test)

