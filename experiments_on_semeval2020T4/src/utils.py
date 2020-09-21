#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from tqdm import tqdm


def get_cache_file_name(file):
    prefix, prev_file_name = os.path.split(file)
    return os.path.join(prefix, os.path.splitext(prev_file_name)[0]+'.p')


def write_cache_word_vectors(file, data):
    with open(get_cache_file_name(file), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def load_cache_word_vectors(file):
    with open(get_cache_file_name(file), 'rb')as pickle_file:
        return pickle.load(pickle_file)


def load_word_vectors(file, dim=300):
    try:
        cache = load_cache_word_vectors(file)
        print("Load from cache")
        return cache
    except FileNotFoundError:
        print("No cache file!")

    if os.path.exists(file):
        word2idx = {}
        idx2word = {}
        embeddings = [np.zeros(dim)]
        word2idx['<padding>'] = 0
        idx2word[0] = '<padding>'

        with open(file, 'r') as fin:
            for i, line in tqdm(enumerate(fin)):
                values = line.split(' ')
                word = values[0]
                idx = i + 1

                word2idx[word] = idx
                idx2word[idx] = word
                embeddings.append(np.asarray(values[1:], dtype='float32'))

            if '<unk>' not in word2idx:
                word2idx['<unk>'] = len(word2idx)
                idx2word[len(idx2word)] = '<unk>'
                embeddings.append(
                    np.random.uniform(low=-0.05, high=0.05, size=dim)
                )

            embeddings = np.asarray(embeddings, dtype='float32')
            write_cache_word_vectors(file, (word2idx, idx2word, embeddings))
            return word2idx, idx2word, embeddings
    else:
        print("{} not found!".format(file))
        raise OSError


if __name__ == '__main__':
    load_word_vectors('/Users/yuhou/Research/hope_git/glove.6B/glove.6B.300d.txt')
