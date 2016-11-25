'''Model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


class MemN2N(object):
    '''End-to-End model'''

    def __init__(self, vocab_size, embedding_size, sentence_length, memory_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.sentence_length = sentence_length
        self.memory_size = memory_size

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
            self.A = tf.get_variable('A', [self.vocab_size, self.embedding_size], tf.float32)
            self.B = tf.get_variable('B', [self.vocab_size, self.embedding_size], tf.float32)
            self.C = tf.get_variable('C', [self.vocab_size, self.embedding_size], tf.float32)

            self.W = tf.get_variable('W', [self.embedding_size, self.vocab_size], tf.float32)

    def _create_inference(self):
        with tf.variable_scope('model'):
            self.memory_input = tf.reduce_sum(tf.nn.embedding_lookup(self.A, self.x, name='m_i_pre'), reduction_indices=[2], name='m_i')
            self.memory_output = tf.reduce_sum(tf.nn.embedding_lookup(self.C, self.x, name='c_i_pre'), reduction_indices=[2], name='c_i')
            self.u = tf.reduce_sum(tf.nn.embedding_lookup(self.B, self.q, name='u_pre'), reduction_indices=[1], name='u', keep_dims=True)

            self.probs = tf.reduce_sum(tf.mul(self.u, self.memory_input, name='u-m_i'), reduction_indices=[2], name='probs', keep_dims=True)
            self.softmax_probs = tf.nn.softmax(self.probs, name='softmax_probs')

            self.o = tf.reduce_sum(tf.mul(self.memory_output, self.softmax_probs, name='p-c_i'), reduction_indices=[1], name='o')
            self.logits = tf.matmul(tf.squeeze(self.u) + self.o, self.W, name='logits')

            self.predicted = tf.argmax(self.logits, 1)


    def _create_loss(self):
        with tf.variable_scope('loss'):
            self.a_one_hot = tf.one_hot(self.a, self.vocab_size, name='answer_one_hot')
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.a_one_hot, name='loss'))

    def _create_train(self):
        with tf.variable_scope('traing'):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
