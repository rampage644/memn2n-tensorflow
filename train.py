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
tf.app.flags.DEFINE_integer('sentence_length', 0, 'Sentence length. Provide to redefine automatically calculated (max would be taken).')
tf.app.flags.DEFINE_integer('memory_size', 50, 'Memory size. Provide to redefine automatically calculated (min would be taken).')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task number to test and train or (in case of independent train)')
tf.app.flags.DEFINE_integer('epoch', 1, 'Epoch count')
tf.app.flags.DEFINE_integer('anneal_every', 10, 'Anneal (halve) learning rate every `anneal_every` epoch')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('hops', 3, 'Hops (layers) count')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Starting learning rate')
tf.app.flags.DEFINE_string('train_dir', os.getcwd(), 'Directory with training files')
tf.app.flags.DEFINE_string('log_dir', os.getcwd(), 'Directory for tensorboard logs')
tf.app.flags.DEFINE_string('ckpt_dir', os.getcwd(), 'Directory for saving/restoring checkpoints')
tf.app.flags.DEFINE_boolean('pe', False, 'Enable position encoding')
tf.app.flags.DEFINE_boolean('joint', False, 'Train model jointly (that is on all tasks instead of one).')


plt.style.use('fivethirtyeight')


def main(argv=None):
    word2idx, idx2word = memn2n.util.load_vocabulary(FLAGS.train_dir)
    if FLAGS.joint:
        train = []
        for task_id in range(1, 21):
            train_task, test_task = memn2n.util.load_dataset_for(task_id, FLAGS.train_dir)
            train.extend(train_task)
            train.extend(test_task)
        train_task, test_task = memn2n.util.load_dataset_for(FLAGS.task_id, FLAGS.train_dir)
        test = list(train_task) + list(test_task)
    else:
        train, test = memn2n.util.load_dataset_for(FLAGS.task_id, FLAGS.train_dir)
        data = list(train) + list(test)
        # keep 10% for validation
        train_size = int((1 - 0.1) * len(data))
        train, test = data[:train_size], data[train_size:]

    memory_size = min(
        memn2n.util.calc_memory_capacity_for(train),
        FLAGS.memory_size
    )

    sentence_length = max(
        memn2n.util.calc_sentence_length_for(train),
        FLAGS.sentence_length
    )

    mem_train, query_train, answer_train = memn2n.util.vectorize_dataset(train, word2idx, memory_size, sentence_length)
    mem_test, query_test, answer_test = memn2n.util.vectorize_dataset(test, word2idx, memory_size, sentence_length)


    with tf.Session() as sess:
        steps_per_epoch = len(mem_train) // FLAGS.batch_size + 1
        print('Model details:')
        for (name, value) in (
            ('step per epoch', steps_per_epoch),
            ('epoch', FLAGS.epoch),
            ('anneal every', FLAGS.anneal_every),
            ('position encoding', FLAGS.pe),
            ('hops', FLAGS.hops),
            ('learning_rate', FLAGS.learning_rate),
            ('vocab_size', len(word2idx)),
            ('embdding size', FLAGS.embedding_size),
            ('sentence length', sentence_length),
            ('memory size', memory_size)
        ):
            print('{}: {}'.format(name, value))

        model = memn2n.model.MemN2N(
            steps_per_epoch,
            FLAGS.epoch,
            FLAGS.anneal_every,
            FLAGS.pe,
            FLAGS.hops,
            FLAGS.learning_rate,
            len(word2idx),
            FLAGS.embedding_size,
            sentence_length,
            memory_size
        )
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter(FLAGS.log_dir, graph=tf.get_default_graph())

        saved_model = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        if saved_model:
            saver.restore(sess, saved_model)
        else:
            print('Prevous model not found, starting from scratch.')
        if not os.path.exists(FLAGS.ckpt_dir):
            os.makedirs(FLAGS.ckpt_dir)

        loss_history = []
        accuracy_history = []
        t = []

        for e in range(FLAGS.epoch):
            for step in range(0, len(mem_train), FLAGS.batch_size):
                # FIXME: last batch size should not to be less than `batch_size`
                start, end = step, step+FLAGS.batch_size if step + FLAGS.batch_size < len(mem_train) else None
                loss, predicted, summary, _ = sess.run([model.loss, model.predicted, model.summary_op, model.train_op], {
                    model.x: mem_train[start:end],
                    model.q: query_train[start:end],
                    model.a: answer_train[start:end]
                })

                loss_history.append(loss)
                t.append(tf.train.global_step(sess, model.global_step))
                writer.add_summary(summary)

            accuracy_history.append(np.array([
                sess.run(model.accuracy, {
                    model.x: mem_train[start:end],
                    model.q: query_train[start:end],
                    model.a: answer_train[start:end]}),
                sess.run(model.accuracy, {
                    model.x: mem_test,
                    model.q: query_test,
                    model.a: answer_test})
            ]))

            print('\rEpoch: {}/{}'.format(e+1, FLAGS.epoch), end='')


        saver.save(sess, os.path.join(FLAGS.ckpt_dir, 'memn2n'), global_step=model.global_step)

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
