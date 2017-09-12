import os
import json
import pickle
import collections
import numpy as np
import random
from config import Config
# import tensorflow as tf
import argparse

# preprocess for language model
class Prepro():
    def __init__(self, vocab_size=3000, len_sent=120):
        self._vocab_size = vocab_size
        self._len_sent = len_sent
        self.config = Config
        self.train_dict = {}
        self.test_dict = {}

    def load_data(self, path):
        features_dict = {}
        for f in os.listdir(path):
            # shape = 80 x 4096
            features = np.load(os.path.join(path, f))
            features_dict[f[:-4]] = features

    def load_labels(self, path, split_ratio=0.8):
        print("load training")
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
                tok = tok[:self.config.cap_length-1]
                tok += ['<eos>']

                caps += [tok]
            cnt_dict[tid] = caps

        print("counting")
        vocabs = cnt.most_common(self.vocab_size - n_res_tok)
        wds, _ = zip(*vocabs)
        for idx, wd in enumerate(wds):
            tokens[wd] = idx + n_res_tok
            revtokens += [wd]

        self._vocab_size = len(tokens)
        self._tokens = tokens

        train_vid_list = []
        train_cap_list = []
        for key, cap in cnt_dict.items():
            for sent in cap:
                s = []
                for wd in sent:
                    s += [tokens[wd] if wd in tokens else tokens['<unk>']]
                train_vid_list += [key]
                train_cap_list += [s]

        n_train = len(train_vid_list)
        n_split = int(n_train * split_ratio)
        dev_vid_list = train_vid_list[n_split:]
        dev_cap_list = train_cap_list[n_split:]

        train_vid_list = train_vid_list[:n_split]
        train_cap_list = train_cap_list[:n_split]

        del cnt_dict
        del cnt

        # test labels
        with open(os.path.join(path, "testing_public_label.json")) as f:
            test_label = json.load(f)

        test_vid_list = []
        test_cap_list = []
        for i in test_label:
            tid = i["id"]
            caption = i["caption"]
            for sent in caption:
                tok = sent[:-1].split(' ')
                tok = tok[:self.config.cap_length-1]
                tok += ['<eos>']
                s = []
                for wd in tok:
                    s += [tokens[wd] if wd in tokens else tokens['<unk>']]
                test_vid_list += [tid]
                test_cap_list += [s]

        train_dict = {'id': train_vid_list, 'caption': train_cap_list}
        dev_dict = {'id': dev_vid_list, 'caption': dev_cap_list}
        test_dict = {'id': test_vid_list, 'caption': test_cap_list}

        self.train_dict = train_dict
        self.dev_dict = dev_dict
        self.test_dict = test_dict
        # return train_dict, test_dict

    def dump_result(self, dest_dir):
        with open(dest_dir + "/train_label.json", "w") as f:
            f.write(json.dumps(self.train_dict))

        with open(dest_dir + "/dev_label.json", "w") as f:
            f.write(json.dumps(self.dev_dict))

        with open(dest_dir + "/test_label.json", "w") as f:
            f.write(json.dumps(self.test_dict))

        with open(dest_dir + "/param.plk", "w") as f:
            pickle.dump(self._tokens, f)
            vals = {'vocab_size': self._vocab_size,
                    'max_length': self._len_sent}
            pickle.dump(vals, f)

    def preprocess(self, src_dir, dest_dir):
        self.load_labels(src_dir)
        self.dump_result(dest_dir)

    @property
    def vocab_size(self):
        return self._vocab_size

def test():
    p = Prepro()
    p.preprocess("a2_data", "test")

def run():
    p = Prepro()
    p.preprocess("MLDS_hw2_data", "preprocessed")

if __name__ == "__main__":
    # test()
    run()