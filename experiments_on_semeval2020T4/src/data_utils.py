#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import pandas
import numpy as np
import nltk

from utils import load_word_vectors


class SemEval20Dataset:
    def __init__(self, data_path, max_seq_len=25):
        self.data_path = data_path
        self.data, self.labels = self.parse()
        self.max_seq_len = max(self.get_max_length(self.data), max_seq_len)

    def parse(self):
        all_data = pandas.read_csv(self.data_path)
        data_dict = []
        for i in range(len(all_data)):
            data_dict.append({'sent': all_data['Correct Statement'][i].lower(), 'label': 1})
            data_dict.append({'sent': all_data['Incorrect Statement'][i].lower(), 'label': 0})
        random.seed(0)
        random.shuffle(data_dict)

        seqs = [nltk.word_tokenize(pair['sent']) for pair in data_dict]
        labels = [pair['label'] for pair in data_dict]

        assert len(labels) == len(seqs)
        # print(seqs)
        return seqs, labels

    @staticmethod
    def get_max_length(seqs):
        max_len = 0
        for seq in seqs:
            max_len = max(len(seq), max_len)
        return max_len

    def get_embeddings_all(self):
        inputs = []
        glove = '/Users/yuhou/Research/hope_git/glove.6B/glove.6B.300d.txt'
        word2idx, _, embeddings = load_word_vectors(glove, 300)
        for seq in self.data:
            word_idx = [0] * self.max_seq_len
            for idx, word in enumerate(seq):
                if word not in word2idx:
                    word_idx[idx] = word2idx['<unk>']
                else:
                    word_idx[idx] = word2idx[word]
            word_vec = [embeddings[i] for i in word_idx]    # (25, 300)
            sent_vec = np.average(word_vec, axis=0)
            inputs.append(sent_vec)
        inputs = np.asarray(inputs, dtype="float32")
        print(inputs.shape)
        return inputs


if __name__ == '__main__':
    file_path = '../dataset/dev.csv'
    dataset = SemEval20Dataset(file_path)
    dataset.get_embeddings_all()
