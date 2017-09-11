import os
import json
import pickle
import collections
import numpy as np
import tensorflow as tf
import argparse

# preprocess for language model
class Prepro():
    def __init__(self, vocab_size=12000, min_len=4, len_sent=120):
        self._vocab_size = vocab_size
        self._len_sent = len_sent
        self._min_len_sent = min_len

    def data_iteration(self):
        for file in os.listdir(self._datadir):
            with open(os.path.join(self._datadir, file)) as f:
                yield f.read().strip().split('.')

    def load_data(self):
        data_generator = self.data_iteration()

        # for data in data_generator:
        cnt = collections.Counter()
        n_res_tok = 4
        tokens = {'<pad>': 0, '<go>': 1, '<eos>': 2, '<unk>': 3}
        revtokens = ['<pad>', '<go>', '<eos>', '<unk>']
        tokensfreq = {}

        self._data = []
        for data in data_generator:
            for sent in data:
                wds = tf.contrib.keras.preprocessing.text.text_to_word_sequence(text=sent,
                                                                                filters='\'"?!.#$%&()*+,-/:;<=>@[\\]^`{|}~\t\n\r')
                if len(wds) > self._min_len_sent:
                    # wds += ['<eos>']
                    cnt.update(wds)
                    self._data += [wds]

        vocabs = cnt.most_common(self._vocab_size - n_res_tok)
        for idx, wd_info in enumerate(vocabs):
            wd, wd_freq = wd_info
            tokens[wd] = idx + n_res_tok
            revtokens += [wd]
            tokensfreq[wd] = wd_freq

        self._vocab_size = len(tokens)

        self._tokens = tokens
        self._revtokens = revtokens

        self._file_word_id = []
        for seq in self._data:
            tmp = []
            seq_tmp = ['<go>'] + seq + ['<eos>']
            for wd in seq_tmp:
                if wd in tokens:
                    tmp += [tokens[wd]]
                else:
                    tmp += [tokens['<unk>']]
            self._file_word_id += [tmp]

        self._pad_seq = tf.contrib.keras.preprocessing.sequence.pad_sequences(self._file_word_id, maxlen=self._len_sent,
                                                                              padding='post')

    def split_data(self):
        train_size = int(self._pad_seq.shape[0] * 0.8)

        self.train_data = self._pad_seq[:train_size]
        self.dev_data = self._pad_seq[train_size:]

        self.train_size = train_size
        self.dev_size = int(self._pad_seq.shape[0]) - train_size

    def dump_result(self):
        with open(self._dest_dir + "/train.json", "w") as f:
            result = {
                'n_samples': self.train_size,
                'data': self.train_data.tolist()
            }
            f.write(json.dumps(result))

        with open(self._dest_dir + "/dev.json", "w") as f:
            result = {
                'n_samples': self.dev_size,
                'data': self.dev_data.tolist()
            }
            f.write(json.dumps(result))

        with open(self._dest_dir + "/param.plk", "w") as f:
            pickle.dump(self._tokens, f)
            vals = {'vocab_size': self._vocab_size,
                    'max_length': self._len_sent}
            pickle.dump(vals, f)

    def preprocess(self, datadir, dest_dir):
        self._datadir = datadir
        self._dest_dir = dest_dir
        print "load data"
        self.load_data()
        print "split data"
        self.split_data()
        print "dump result"
        self.dump_result()

    def preprocess_test(self, dest_dir):
        print "preprocess test"
        import pandas as pd
        with open(os.path.join(dest_dir, "param.plk"), "r") as f:
            vocab = pickle.load(f)
            vals = pickle.load(f)

        len_sent = vals['max_length']
        tokens = vocab
        test_table = pd.read_csv(os.path.join("hw1_data", "testing_data.csv"))
        ans_token = "_____"
        questions = test_table['question']
        candidates = zip(test_table['a)'], test_table['b)'], test_table['c)'], test_table['d)'], test_table['e)'])

        data = []
        for idx, q in enumerate(questions):
            l = tf.contrib.keras.preprocessing.text.text_to_word_sequence(text=q,
                                                                          filters='\'"?!.#$%&()*+,-/:;<=>@[\\]^`{|}~\t\n\r')
            l = ['<go>'] + l + ['<eos>']
            i = l.index(ans_token)
            for cand in candidates[idx]:
                tmp = []
                l[i] = cand
                for wd in l:
                    if wd in tokens:
                        tmp += [tokens[wd]]
                    else:
                        tmp += [tokens['<unk>']]
                data += [tmp]

        data = tf.contrib.keras.preprocessing.sequence.pad_sequences(data, maxlen=len_sent, padding='post')

        with open(dest_dir + "/test.json", "w") as f:
            result = {
                'n_samples': data.shape[0],
                'data': data.tolist()
            }
            # pickle.dump(result, f)
            f.write(json.dumps(result))

def main():
    p = Prepro()
    # p.preprocess("hw1_data/Holmes_Training_Data", "processed")
    p.preprocess_test("processed")

if __name__ == "__main__":
    main()
