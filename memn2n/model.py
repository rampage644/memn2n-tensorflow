
'''Model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


class MemN2N(object):
    '''End-to-End model'''

    def __init__(self, lr, vocab_size, embedding_size, sentence_length, memory_size):
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

        with tf.variable_scope('input'):
            # sentences - stories - facts
            self.x = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_length], name='facts')
            self.q = tf.placeholder(tf.int32, [None, self.sentence_length], name='query')
            self.a = tf.placeholder(tf.int32, [None], name='answer')

        with tf.variable_scope('embeddings'):
            # +1 is for fixed zero input
            zero_embedding = tf.zeros([1, self.embedding_size])

            self.A = tf.concat(0, [
                zero_embedding,
                tf.get_variable('A', [self.vocab_size, self.embedding_size], tf.float32),
            ])
            self.TA = tf.get_variable('TA', [self.memory_size, self.embedding_size], tf.float32)

            self.B = tf.concat(0, [
                zero_embedding,
                tf.get_variable('B', [self.vocab_size, self.embedding_size], tf.float32)
            ])
            self.C = tf.concat(0, [
                zero_embedding,
                tf.get_variable('C', [self.vocab_size, self.embedding_size], tf.float32)
            ])
            self.TC = tf.get_variable('TC', [self.memory_size, self.embedding_size], tf.float32)

            self.W = tf.concat(1, [
                tf.get_variable('W', [self.embedding_size, self.vocab_size], tf.float32),
                tf.transpose(zero_embedding)
            ])

            tf.histogram_summary('A', self.A)
            tf.histogram_summary('B', self.B)
            tf.histogram_summary('C', self.C)
            tf.histogram_summary('W', self.W)

    def _create_inference(self):
        with tf.variable_scope('model'):
            self.memory_input = tf.reduce_sum(tf.nn.embedding_lookup(self.A, self.x, name='m_i_pre'), reduction_indices=[2], name='m_i') + self.TA
            self.memory_output = tf.reduce_sum(tf.nn.embedding_lookup(self.C, self.x, name='c_i_pre'), reduction_indices=[2], name='c_i') + self.TC
            self.u = tf.reduce_sum(tf.nn.embedding_lookup(self.B, self.q, name='u_pre'), reduction_indices=[1], name='u', keep_dims=True)

            self.probs = tf.reduce_sum(tf.mul(self.u, self.memory_input, name='u-m_i'), reduction_indices=[2], name='probs', keep_dims=True)
            self.softmax_probs = tf.nn.softmax(self.probs, name='softmax_probs', dim=1)

            self.o = tf.reduce_sum(tf.mul(self.memory_output, self.softmax_probs, name='p-c_i'), reduction_indices=[1], name='o')
            self.logits = tf.matmul(tf.squeeze(self.u) + self.o, self.W, name='logits')

            self.predicted = tf.argmax(tf.nn.softmax(self.logits), 1)


    def _create_loss(self):
        with tf.variable_scope('loss'):
            a_one_hot = tf.one_hot(self.a, self.vocab_size + 1, name='answer_one_hot')
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.logits, a_one_hot, name='loss'))

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
