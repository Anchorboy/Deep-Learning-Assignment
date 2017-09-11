import os
import json
import pickle
import collections
import numpy as np
import tensorflow as tf
import argparse

# preprocess for language model
class Prepro():
    def __init__(self, vocab_size=3000, len_sent=120):
        self._vocab_size = vocab_size
        self._len_sent = len_sent
        self.train_dict = {}
        self.test_dict = {}

    def load_data(self, path):
        features_dict = {}
        for f in os.listdir(path):
            # shape = 80 x 4096
            features = np.load(os.path.join(path, f))
            features_dict[f[:-4]] = features

    def load_labels(self, path):
        print "load training"
        # training labels
        with open(os.path.join(path, "training_label.json")) as f:
            train_label = json.load(f)

        # calculate word count
        cnt = collections.Counter()
        n_res_tok = 4
        tokens = {'<pad>':0, '<bos>':1, '<eos>':2, '<unk>':3}
        revtokens = ['<pad>', '<bos>', '<eos>', '<unk>']

        cnt_dict = {}
        for i in train_label:
            tid = i["id"]
            caps = []
            caption = i["caption"]
            for sent in caption:
                tok = sent[:-1].split(' ')
                cnt.update(tok)
                tok += ['<eos>']

                caps += [tok]
            cnt_dict[tid] = caps

        print "counting"
        vocabs = cnt.most_common(self.vocab_size - n_res_tok)
        wds, _ = zip(*vocabs)
        for idx, wd in enumerate(wds):
            tokens[wd] = idx + n_res_tok
            revtokens += [wd]

        self._vocab_size = len(tokens)
        self._tokens = tokens

        train_dict = {}
        for key, cap in cnt_dict.iteritems():
            ss = []
            for sent in cap:
                s = []
                for wd in sent:
                    s += [tokens[wd] if wd in tokens else tokens['<unk>']]
                ss += [s]
            train_dict[key] = ss

        del cnt_dict
        del cnt

        # test labels
        with open(os.path.join(path, "testing_public_label.json")) as f:
            test_label = json.load(f)

        test_dict = {}
        for i in test_label:
            tid = i["id"]
            ss = []
            caption = i["caption"]
            for sent in caption:
                tok = sent[:-1].split(' ')
                tok += ['<eos>']
                s = []
                for wd in tok:
                    s += [tokens[wd] if wd in tokens else tokens['<unk>']]
                ss += [s]
            test_dict[tid] = ss

        return train_dict, test_dict

    def dump_result(self, dest_dir):
        with open(dest_dir + "/train_label.json", "w") as f:
            f.write(json.dumps(self.train_dict))

        with open(dest_dir + "/test_label.json", "w") as f:
            f.write(json.dumps(self.test_dict))

        with open(dest_dir + "/param.plk", "w") as f:
            pickle.dump(self._tokens, f)
            vals = {'vocab_size': self._vocab_size,
                    'max_length': self._len_sent}
            pickle.dump(vals, f)

    def preprocess(self, src_dir, dest_dir):
        self.train_dict, self.test_dict = self.load_labels(src_dir)
        self.dump_result(dest_dir)

    @property
    def vocab_size(self):
        return self._vocab_size

def test():
    p = Prepro()
    p.preprocess("a2_data", "preprocessed")

if __name__ == "__main__":
    test()