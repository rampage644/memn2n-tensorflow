# End-To-End Memory Networks

Replicate study for [End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v5.pdf) paper.

Still to-do (see Issues):

 * [ ] Linear start
 * [ ] Random noise
 * [ ] Langauage model

## Usage
```
$ python train.py --help
usage: train.py [-h] [--embedding_size EMBEDDING_SIZE]
                [--sentence_length SENTENCE_LENGTH]
                [--memory_size MEMORY_SIZE] [--task_id TASK_ID]
                [--epoch EPOCH] [--batch_size BATCH_SIZE] [--hops HOPS]
                [--learning_rate LEARNING_RATE] [--train_dir TRAIN_DIR]
                [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR] [--pe [PE]] [--nope]
                [--joint [JOINT]] [--nojoint]

optional arguments:
  -h, --help            show this help message and exit
  --embedding_size EMBEDDING_SIZE
                        Dimension for word embedding
  --sentence_length SENTENCE_LENGTH
                        Sentence length. Provide to redefine automatically
                        calculated (max would be taken).
  --memory_size MEMORY_SIZE
                        Memory size. Provide to redefine automatically
                        calculated (max would be taken).
  --task_id TASK_ID     Task number to test and train or (in case of
                        independent train)
  --epoch EPOCH         Epoch count
  --batch_size BATCH_SIZE
                        Batch size
  --hops HOPS           Hops (layers) count
  --learning_rate LEARNING_RATE
                        Starting learning rate
  --train_dir TRAIN_DIR
                        Directory with training files
  --log_dir LOG_DIR     Directory for tensorboard logs
  --ckpt_dir CKPT_DIR   Directory for saving/restoring checkpoints
  --pe [PE]             Enable position encoding
  --nope
  --joint [JOINT]       Train model jointly (that is on all tasks instead of
                        one).
  --nojoint
```

For example, to train and test model with 5th bAbI task use
```
python train.py --train_dir data/tasks_1-20_v1-2/en/ --epoch 15 --learning_rate 0.01 --task_id 5 --hops 3
```

To train model jointly use `--joint` flag. Model would be trained on all bAbI tasks (1th-20th) and tested on given with `--task_id` task.
```
python train.py --train_dir data/tasks_1-20_v1-2/en/ --epoch 15 --learning_rate 0.01 --task_id 5 --hops 3 --joint
```

## Datasets

* [bAbI](http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz)

## Other implementations

* https://github.com/facebook/MemNN (matlab and torch, original source)
* https://github.com/carpedm20/MemN2N-tensorflow
* https://github.com/domluna/memn2n
* https://github.com/seominjoon/memnn-tensorflow
* https://github.com/vinhkhuc/MemN2N-babi-python

