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
tf.app.flags.DEFINE_integer('embedding_size', 15, 'Dimension of word embedding')
tf.app.flags.DEFINE_integer('sentence_length', 0, 'Sentence length')
tf.app.flags.DEFINE_integer('memory_size', 0, 'Max memory size')
tf.app.flags.DEFINE_integer('task_id', 1, 'Task id to train')
tf.app.flags.DEFINE_integer('epoch', 1, 'Epoch number')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.app.flags.DEFINE_string('train_dir', os.getcwd(), 'Directory with training files')
tf.app.flags.DEFINE_string('log_dir', os.getcwd(), 'Directory for tensorboard logs')


plt.style.use('fivethirtyeight')


def main(argv=None):
    word2idx, idx2word = memn2n.util.load_vocabulary(FLAGS.train_dir)
    train, test = memn2n.util.load_dataset_for(FLAGS.task_id, FLAGS.train_dir)
    data = list(train) + list(test)
    # keep 10% for validation
    train_size = int((1 - 0.1) * len(data))
    train, test = data[:train_size], data[train_size:]

    memory_size = max(
        memn2n.util.calc_memory_capacity_for(train),
        memn2n.util.calc_memory_capacity_for(test),
        FLAGS.memory_size
    )

    sentence_length = max(
        memn2n.util.calc_sentence_length_for(train),
        memn2n.util.calc_sentence_length_for(test),
        FLAGS.sentence_length
    )

    mem_train, query_train, answer_train = memn2n.util.vectorize_dataset(train, word2idx, memory_size, sentence_length)
    mem_test, query_test, answer_test = memn2n.util.vectorize_dataset(test, word2idx, memory_size, sentence_length)

    with tf.Session() as sess:
        model = memn2n.model.MemN2N(
            FLAGS.learning_rate,
            len(word2idx),
            FLAGS.embedding_size,
            sentence_length,
            memory_size
        )
        writer = tf.train.SummaryWriter(FLAGS.log_dir, graph=tf.get_default_graph())

        sess.run(tf.initialize_all_variables())
        loss_history = []
        accuracy_history = []
        t = []

        for e in range(FLAGS.epoch):
            for step in range(0, len(mem_train), FLAGS.batch_size):
                start, end = step, step+FLAGS.batch_size if step + 2 * FLAGS.batch_size < len(mem_train) else None
                loss, predicted, summary, _ = sess.run([model.loss, model.predicted, model.summary_op, model.train_op], {
                    model.x: mem_train[start:end],
                    model.q: query_train[start:end],
                    model.a: answer_train[start:end]
                })

                loss_history.append(loss)
                t.append(tf.train.global_step(sess, model.global_step))
                writer.add_summary(summary)

            accuracy_history.append(np.array([
                sess.run(model.accuracy, {model.x: mem_train, model.q: query_train, model.a: answer_train}),
                sess.run(model.accuracy, {model.x: mem_test, model.q: query_test, model.a: answer_test})
            ]))

            print('\rEpoch: {}/{}'.format(e+1, FLAGS.epoch), end='')

        accuracy_history = np.asarray(accuracy_history)
        print()
        print('Accuracy train: {}, test: {}'.format(accuracy_history[-1, 0], accuracy_history[-1, 1]))
        _, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title('Loss')
        ax1.plot(t, loss_history)
        ax1.plot(t, np.r_[loss_history[:19], memn2n.util.moving_average(loss_history, n=20)])
        ax2.set_title('Accuracy')
        ax2.plot(accuracy_history[:, 0])
        ax2.plot(accuracy_history[:, 1])
        plt.show()

if __name__ == '__main__':
    tf.app.run()
