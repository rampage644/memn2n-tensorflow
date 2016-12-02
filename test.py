'''Train model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import memn2n.model
import memn2n.util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_size', 15, 'Dimension for word embedding')
tf.app.flags.DEFINE_integer('sentence_length', 10, 'Sentence length. Provide to redefine automatically calculated (max would be taken).')
tf.app.flags.DEFINE_integer('memory_size', 50, 'Memory size. Provide to redefine automatically calculated (min would be taken).')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task number to test and train or (in case of independent train)')
tf.app.flags.DEFINE_integer('hops', 3, 'Hops (layers) count')
tf.app.flags.DEFINE_string('ckpt_dir', os.getcwd(), 'Directory for saving/restoring checkpoints')
tf.app.flags.DEFINE_string('train_dir', os.getcwd(), 'Directory with training files')
tf.app.flags.DEFINE_boolean('pe', False, 'Enable position encoding')


def main(argv=None):
    word2idx, idx2word = memn2n.util.load_vocabulary(FLAGS.train_dir)

    with tf.Session() as sess:
        model = memn2n.model.MemN2N(
            1,
            1,
            1,
            FLAGS.pe,
            FLAGS.hops,
            0.0,
            len(word2idx),
            FLAGS.embedding_size,
            FLAGS.sentence_length,
            FLAGS.memory_size
        )

        saver = tf.train.Saver()
        saved_model_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        if saved_model_path:
            saver.restore(sess, saved_model_path)
        else:
            print('Checkpoint not found, exiting.')
            return 1

        tasks = [FLAGS.task_id] if FLAGS.task_id else range(1, 21)
        for task_id in tasks:
            _, data = memn2n.util.load_dataset_for(task_id, FLAGS.train_dir)
            mem, query, answer = memn2n.util.vectorize_dataset(
                list(data), word2idx, FLAGS.memory_size, FLAGS.sentence_length)

            accuracy = sess.run(model.accuracy, {
                model.x: mem, model.q: query, model.a: answer
            })

            print('{}: {:.2f}'.format(task_id, accuracy))


if __name__ == '__main__':
    tf.app.run()
