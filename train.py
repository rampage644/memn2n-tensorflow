'''Train model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf

from memn2n.model import MemN2N
from memn2n.util import create_vocabulary, load_dataset_for


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_size', 500, 'Dimension of word embedding')
tf.app.flags.DEFINE_integer('sentence_length', 50, 'Sentence length')
tf.app.flags.DEFINE_string('train_dir', os.getcwd(), 'Directory with training files')
tf.app.flags.DEFINE_integer('task_id', 1, 'Task id to train')


def main(argv=None):
    vocab = create_vocabulary(FLAGS.train_dir)
    train, test = load_dataset_for(FLAGS.task_id, FLAGS.train_dir)

    with tf.Session() as sess:
        model = MemN2N(len(vocab), FLAGS.embedding_size, FLAGS.sentence_length)

        sess.run(tf.initialize_all_variables())

        loss, _ = sess.run([model.loss, model.train_op], {
            model.x: train,
            model.q: [],
            model.a: []
        })

if __name__ == '__main__':
    tf.app.run()