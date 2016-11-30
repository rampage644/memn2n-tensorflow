'''Model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

#pylint: disable=C0103
class MemN2N(object):
    '''End-to-End model'''

    def __init__(self, use_pe, hops, lr, vocab_size, embedding_size, sentence_length, memory_size):
        self.use_pe = use_pe
        self.hops = hops
        self.lr = lr
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.sentence_length = sentence_length
        self.memory_size = memory_size

        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_train()
        self._create_accuracy()

        self.summary_op = tf.merge_all_summaries()

    def _encoding(self, embeddings):
        return tf.mul(self.l_pe, embeddings) if self.use_pe else embeddings

    def _create_variables(self):
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES]
        )

        steps_per_epoch = 500
        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [steps_per_epoch * 3, steps_per_epoch * 6, steps_per_epoch * 9],
            [self.lr, self.lr / 2, self.lr / 4, self.lr / 8]
        )

        J, D = self.sentence_length, self.embedding_size
        j = np.expand_dims(np.linspace(1, J, J), 0)
        k = np.expand_dims(np.linspace(1, D, D), 1)
        l = (1 - j / J) - k / D * (1 - 2 * j / J)
        self.l_pe = tf.constant(l, tf.float32, shape=[J, D])

        with tf.variable_scope('input'):
            # sentences - stories - facts
            self.x = tf.placeholder(tf.int32,
                                    [None, self.memory_size, self.sentence_length],
                                    name='facts')
            self.q = tf.placeholder(tf.int32, [None, self.sentence_length], name='query')
            self.a = tf.placeholder(tf.int32, [None], name='answer')

        self.embeddings = {}
        with tf.variable_scope('embeddings'):
            # +1 is for fixed zero input
            zero_embedding = tf.zeros([1, self.embedding_size])

            B = tf.concat(0, [
                zero_embedding,
                tf.get_variable('B', [self.vocab_size, self.embedding_size], tf.float32),
            ])
            self.embeddings['B'] = B
            C_prev = B
            TC_prev = tf.get_variable('TB', [self.memory_size, self.embedding_size], tf.float32)
            for k in range(1, self.hops + 1):
                k = str(k)

                with tf.variable_scope('hop' + k):
                    C = tf.concat(0, [
                        zero_embedding,
                        tf.get_variable('C', [self.vocab_size, self.embedding_size], tf.float32)
                    ])
                    TC = tf.get_variable('TC', [self.memory_size, self.embedding_size], tf.float32)

                self.embeddings['A' + k] = C_prev
                self.embeddings['TA' + k] = TC_prev
                self.embeddings['C' + k] = C
                self.embeddings['TC' + k] = TC

                C_prev = C
                TC_prev = TC

                tf.histogram_summary('A' + k, self.embeddings['A' + k])
                tf.histogram_summary('C' + k, C)

            self.embeddings['W'] = tf.transpose(self.embeddings['C' + str(self.hops)])

    def _create_inference(self):
        with tf.variable_scope('model'):
            B, W = self.embeddings['B'], self.embeddings['W']
            u_prev = tf.reduce_sum(self._encoding(tf.nn.embedding_lookup(B, self.q, name='u_pre')),
                                   reduction_indices=[1], name='u', keep_dims=True)

            for k in range(1, self.hops + 1):
                k = str(k)
                A, C, TA, TC = (
                    self.embeddings['A' + k], self.embeddings['C' + k],
                    self.embeddings['TA' + k], self.embeddings['TC' + k]
                )

                with tf.variable_scope('hop' + k):
                    m = tf.reduce_sum(
                        self._encoding(tf.nn.embedding_lookup(A, self.x, name='m_i_pre')),
                        reduction_indices=[2], name='m_i') + TA
                    c = tf.reduce_sum(
                        self._encoding(tf.nn.embedding_lookup(C, self.x, name='c_i_pre')),
                        reduction_indices=[2], name='c_i') + TC

                    p = tf.reduce_sum(tf.mul(u_prev, m, name='u-m_i'),
                                      reduction_indices=[2], name='probs', keep_dims=True)
                    softmax_p = tf.nn.softmax(p, name='softmax_probs', dim=1)

                    o = tf.reduce_sum(tf.mul(c, softmax_p, name='p-c_i'),
                                      reduction_indices=[1], name='o', keep_dims=True)

                    u = u_prev + o
                    u_prev = u

            self.logits = tf.matmul(tf.squeeze(u), W, name='logits')
            self.predicted = tf.argmax(tf.nn.softmax(self.logits), 1)


    def _create_loss(self):
        with tf.variable_scope('loss'):
            a_one_hot = tf.one_hot(self.a, self.vocab_size + 1, name='answer_one_hot')
            self.loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, a_one_hot, name='loss'))

            tf.scalar_summary('Loss', self.loss)

    def _create_accuracy(self):
        with tf.variable_scope('accuracy'):
            correct = tf.equal(tf.cast(self.predicted, tf.int32), self.a)
            self.accuracy = 100.0 * tf.reduce_mean(tf.cast(correct, tf.float32))

            tf.scalar_summary('accuracy', self.accuracy)

    def _create_train(self):
        with tf.variable_scope('training'):
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            gvs = train_op.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(grad, 40.0), var) for grad, var in gvs]
            self.train_op = train_op.apply_gradients(capped_gvs, global_step=self.global_step)
