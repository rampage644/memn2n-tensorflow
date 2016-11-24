'''Model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


class MemN2N(object):

    def __init__(self, vocab_size, embedding_size, sentence_length):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.sentence_length = sentence_length

        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_train()

    def _create_variables(self):
        with tf.variable_scope('input'):
            # sentences - stories - facts
            self.x = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_length], name='facts')
            self.q = tf.placeholder(tf.int32, [None, self.sentence_length], name='query')
            self.a = tf.placeholder(tf.int32, [None], name='answer')

        with tf.variable_scope('embeddings'):
            self.A = tf.get_variable([None, self.vocab_size, self.embedding_size])
            self.B = tf.get_variable([None, self.vocab_size, self.embedding_size])
            self.C = tf.get_variable([None, self.vocab_size, self.embedding_size])

            self.W = tf.get_variable([None, self.embedding_size, self.vocab_size])

    def _create_inference(self):
        with tf.variable_scope('model'):
            self.memory_input = tf.nn.embedding_lookup(self.x, self.A)
            self.memory_output = tf.nn.embedding_lookup(self.x, self.C)

            self.u = tf.nn.embedding_lookup(self.q, self.B)
            self.probs = tf.matmul(self.u, self.memory_input, transpose_a=True)

            self.o = tf.matmul(self.probs, self.memory_output)
            self.logits = tf.matmul(self.W, self.o + self.u)

    def _create_loss(self):
        with tf.variable_scope('loss'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.a)

    def _create_train(self):
        with tf.variable_scope('traing'):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
