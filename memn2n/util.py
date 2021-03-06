'''Utility functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import numpy as np
from nltk.tokenize import word_tokenize as tokenize


def load_dataset(filename):
    '''Load bAbI dataset from given file'''


    with open(filename) as ifile:
        current_id = 0

        facts = []
        for line in ifile:
            fact_or_query, answer, *_ = line.split('\t') + [None] * 2

            _id, *words = tokenize(fact_or_query)
            _id = int(_id)
            fact_or_query = ' '.join(words)

            if _id < current_id:
                facts = []
            current_id = _id

            if answer:
                yield (facts[:], fact_or_query, answer)
            else:
                facts.append(fact_or_query)


def load_dataset_for(task, directory):
    '''Load train and test datasets for task_id = task'''

    train_set_filename = [f for f in os.listdir(directory) if 'qa' + str(task) + '_' in f and 'train' in f][0]
    test_set_filename = [f for f in os.listdir(directory) if 'qa' + str(task) + '_' in f and 'test' in f][0]

    return (
        load_dataset(os.path.join(directory, train_set_filename)),
        load_dataset(os.path.join(directory, test_set_filename))
    )


def create_vocabulary(directory):
    tokens = [token for f in os.listdir(directory)
              for token in tokenize(open(os.path.join(directory, f)).read())
              if not token.isdigit()]
    return {v: k for k, v in enumerate(set(tokens), start=1)}


def vectorize_dataset(dataset, word2idx, memory_size, sentence_length):
    def word2idx_func(x):
        return word2idx.get(x, 0)

    def pad_2d_to(width, array):
        d1, d2 = abs(width[0] - array.shape[0]), abs(width[1] - array.shape[1])
        return np.pad(array, ((0, d1), (0, d2)), 'constant')

    def pad_1d_to(width, array):
        d = abs(width - array.shape[0])
        return np.pad(array, ((0, d)), 'constant')

    N = len(dataset)
    facts = np.zeros((N, memory_size, sentence_length))
    query = np.zeros((N, sentence_length))
    answer = np.zeros((N))
    for idx, (fcts, q, a) in enumerate(dataset):
        facts[idx] = pad_2d_to([memory_size, sentence_length], np.vstack([pad_1d_to(sentence_length, np.fromiter(map(word2idx_func, tokenize(f)), np.int32)) for f in fcts])[-memory_size:])
        query[idx] = pad_1d_to(sentence_length, np.fromiter(map(word2idx_func, tokenize(q)), np.int32))
        answer[idx] = word2idx_func(a)
    return facts, query, answer


def moving_average(a, n=3) :
    '''Simple moving average function.'''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_vocabulary(directory):
    if os.path.exists(os.path.join(directory, '.memn2n_voc')):
        with open(os.path.join(directory, '.memn2n_voc'), 'rb') as pfile:
            word2idx, idx2word = pickle.load(pfile)
    else:
        word2idx = create_vocabulary(directory)
        idx2word = {v: k for k,v in word2idx.items()}
        with open(os.path.join(directory, '.memn2n_voc'), 'wb') as pfile:
            pickle.dump((word2idx, idx2word), pfile)

    return word2idx, idx2word


def calc_memory_capacity_for(dataset):
    return max(map(lambda x: len(x), map(lambda x: x[0], dataset)))


def calc_sentence_length_for(dataset):
    return max(
        max(len(tokenize(fact)) for sample in dataset for fact in sample[0]),
        max(len(tokenize(sample[1])) for sample in dataset)
    )
