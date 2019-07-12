# -*- coding: utf-8 -*-

import re
import numpy as np
from tqdm import tqdm
from collections import Counter

_FILE_PATH = './data/train.tsv'
_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_WORD_REGEX = re.compile(r"(?u)\b\w\w+\b")
_PUNC_TABLE = str.maketrans("", "", _PUNCTUATION)
_STOP_WORDS = []
with open('./stopwords.txt', 'r') as f:
    _STOP_WORDS = f.readlines()


def tokenize_words(line, lowercase=True, filter_stopwords=True):
    words = _WORD_REGEX.findall(line.lower() if lowercase else line)
    return remove_stop_words(words) if filter_stopwords else words


def remove_stop_words(words):
    return [word for word in words if word not in _STOP_WORDS]


class Vocab(object):
    def __init__(self, fp=_FILE_PATH, mode='train', lowercase=True, filter_punc=True):
        self.datasets = self._read_tsv(fp, mode)
        self.word2idx, self.idx2word = self._word_dict(self.datasets)
        self.mode = mode

    def __len__(self):
        return len(self.word2idx)

    def get_batch(self, batch_size=32):
        lines = [tokenize_words(data[0]) for data in self.datasets]
        batch_num = len(lines) // batch_size
        for batch in range(batch_num):
            voc_mat = [self.line2BowVec(line) for line in lines[batch*batch_size:(batch+1)*batch_size]]
            labels = [data[1] for data in self.datasets[batch*batch_size:(batch+1)*batch_size]] \
                        if self.mode == 'train' else []
            yield np.array(voc_mat), np.array(labels)

    @property
    def labels(self):
        labels = np.array([data[1] for data in self.datasets]) \
            if self.mode == 'train' else np.zeros((1, len(self.datasets)))
        return labels

    @property
    def mat(self):
        lines = [tokenize_words(data[0]) for data in self.datasets]
        voc_mat = [self.line2BowVec(line) for line in tqdm(lines)]
        return np.array(voc_mat)

    def line2BowVec(self, line):
        vec = len(self.word2idx) * [0]
        for k, v in Counter(line).items():
            vec[self.word2idx[k]] = v
        return vec

    def _word_dict(self, datasets):
        words = []
        for dataset in datasets:
            words += tokenize_words(dataset[0])
        words = list(set(words))
        word2idx = {word: i for i, word in enumerate(words)}
        idx2words = {i: word for i, word in enumerate(words)}
        return word2idx, idx2words

    def _read_tsv(self, fp, mode='train'):
        datasets = []
        with open(fp, 'r', encoding='utf-8') as fr:
            for lidx, line in enumerate(fr):
                if lidx == 0:
                    continue
                lines = line.strip().split('\t')
                # print(len(lines))
                if len(lines) == 4 and mode == 'train':
                    phase, label = lines[2], lines[3]
                    datasets.append([phase, int(label)])
                if len(lines) == 3 and mode == 'test':
                    datasets.append(lines[-1])
        return datasets


if __name__ == "__main__":
    vocab = Vocab()
    print(len(vocab))
    for train_data, train_label in vocab.get_batch():
        print(train_data, train_label)
        break

