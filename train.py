'''Train model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf
import numpy as np

import memn2n.model
import memn2n.util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Dimension of word embedding')
tf.app.flags.DEFINE_integer('sentence_length', 20, 'Sentence length')
tf.app.flags.DEFINE_integer('memory_size', 50, 'Max memory size')
tf.app.flags.DEFINE_integer('task_id', 1, 'Task id to train')
tf.app.flags.DEFINE_integer('epoch', 1, 'Epoch number')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_string('train_dir', os.getcwd(), 'Directory with training files')


def main(argv=None):
    word2idx = memn2n.util.create_vocabulary(FLAGS.train_dir)
    idx2word = {v: k for k,v in word2idx.items()}
    train, test = memn2n.util.load_dataset_for(FLAGS.task_id, FLAGS.train_dir)

    mem_train, query_train, answer_train = memn2n.util.vectorize_dataset(list(train), word2idx, FLAGS.memory_size, FLAGS.sentence_length)
    mem_test, query_test, answer_test = memn2n.util.vectorize_dataset(list(test), word2idx, FLAGS.memory_size, FLAGS.sentence_length)

    with tf.Session() as sess:
        model = memn2n.model.MemN2N(len(word2idx), FLAGS.embedding_size, FLAGS.sentence_length, FLAGS.memory_size)

        sess.run(tf.initialize_all_variables())

    for e in range(FLAGS.epoch):
        for step in range(0, len(mem_train), FLAGS.batch_size):
            start, end = step, step+FLAGS.batch_size if step + FLAGS.batch_size < len(mem_train) else None
            loss, predicted, _ = sess.run([model.loss, model.predicted, model.train_op], {
                model.x: mem_train[start:end],
                model.q: query_train[start:end],
                model.a: answer_train[start:end]
            })

            print('loss={}'.format(loss))

        predicted = sess.run(model.predicted, {
            model.x: mem_test,
            model.q: query_test
        })

        print('accuracy={}'.format(100.0 * (predicted == answer_test).astype(np.int32).sum() / len(predicted)))

if __name__ == '__main__':
    tf.app.run()
