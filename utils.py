import math

import tensorflow as tf


class COSINEANNEALINGLR:
    def __init__(self, initial_learning_rate, decay_steps, min_lr):
        self._step = 0
        self._initial_learning_rate = initial_learning_rate
        self._decay_steps = decay_steps
        self._min_lr = min_lr

    def step(self):
        self._step += 1

    def get_lr(self):
        step = min(self._step, self._decay_steps)
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / self._decay_steps))
        decayed_lr = self._initial_learning_rate * cosine_decay
        if decayed_lr < self._min_lr:
            decayed_lr = self._min_lr
        return decayed_lr


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = tf.math.top_k(output, k=maxk)
    pred = tf.transpose(pred)
    correct = tf.math.equal(
        pred,
        tf.cast(tf.broadcast_to(tf.reshape(target, shape=(1, -1)), shape=pred.shape), dtype=tf.int32)
    )

    res = []
    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(tf.reshape(correct[:k], shape=(-1,)), dtype=tf.float32))
        res.append(correct_k * (100.0 / batch_size))
    return res
