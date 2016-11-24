'''Utility functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
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

    train_set_filename = [f for f in os.listdir(directory) if 'qa' + str(task) in f and 'train' in f][0]
    test_set_filename = [f for f in os.listdir(directory) if 'qa' + str(task) in f and 'test' in f][0]

    return (
        load_dataset(os.path.join(directory, train_set_filename)),
        load_dataset(os.path.join(directory, test_set_filename))
    )


def create_vocabulary(directory):
    tokens = [token for f in os.listdir(directory)
              for token in tokenize(open(os.path.join(directory, f)).read())
              if not token.isdigit()]
    return {k: v for k, v in enumerate(set(tokens), start=1)}
