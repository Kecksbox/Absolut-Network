import argparse
import datetime
import logging
import math
import os
from typing import List

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import utils
from utils import COSINEANNEALINGLR

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=99999, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=2, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--cell_steps', type=int, default=4, help='total number of layers')
parser.add_argument('--cell_multiplier', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--log_path', type=str, default='logs', help='path to save tensorboard logs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--val_portion', type=float, default=0.0, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--mixed_op_type', type=str, default='binary', help='the type of mixed operation used in the cells')
parser.add_argument('--mixed_op_temperature', type=float, default=0.5,
                    help='temperature applied to the gumble softmax sampels (binary versions only).')
args = parser.parse_args()


class Madam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, g_bound=10.0,
                 bind_lr: bool = False,
                 name="Madam", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)

        assert learning_rate < 1

        self.step = 0
        self.g_bound = g_bound
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))  # handle lr=learning_rate

        self.bind_lr = bind_lr

        self.lr_t = learning_rate

        self.initial_lr = learning_rate

    def apply_gradients(
        self, grads_and_vars, name=None, experimental_aggregate_gradients=True
    ):

        lr_t = self._decayed_lr(tf.float32)

        grads_and_vars = list(grads_and_vars)

        # TODO: Doesn`t work that well better replace with line search
        if self.bind_lr:
            groups = dict()
            for grad, var in grads_and_vars:
                layer_name = str(var.name.split('/', 1)[0])
                if layer_name not in groups:
                    groups[layer_name] = (tf.reshape(grad, shape=(-1,)), tf.reshape(var, shape=(-1,)))
                else:
                    grad = tf.reshape(grad, shape=(-1,))
                    var = tf.reshape(var, shape=(-1,))
                    groups[layer_name] = (
                        tf.concat([groups[layer_name][0], grad], axis=0),
                        tf.concat([groups[layer_name][1], var], axis=0)
                    )

            L = len(groups)
            trust = self.initial_lr
            lr_t = float('inf')
            for grad, var in groups.values():
                abs_grad = tf.abs(grad)
                abs_var = tf.abs(var)
                dot = tf.reduce_sum(tf.reshape(abs_grad, shape=(-1,)) * tf.reshape(abs_var, shape=(-1,)))
                cos_angle = dot / (tf.norm(abs_grad) * tf.norm(abs_var))
                max_lr = tf.math.pow(1 + cos_angle, 1/L) - 1
                if lr_t >= max_lr:
                    lr_t = trust * max_lr

        self.lr_t = lr_t

        super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "g_")

    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self.lr_t, dtype=var_dtype)

        if grad is None:
            return

        exp_avg_sq = self.get_slot(var, "g_")

        self.step += 1
        bias_correction = 1 - 0.999 ** self.step
        exp_avg_sq.assign(0.999 * exp_avg_sq + 0.001 * grad ** 2)

        g_normed = grad / tf.math.sqrt(exp_avg_sq / bias_correction)
        g_normed_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(g_normed)), dtype=tf.float32)
        g_normed = tf.math.multiply_no_nan(g_normed, g_normed_not_nan)
        g_normed = tf.clip_by_value(g_normed, -self.g_bound, self.g_bound)

        new_var = var * tf.math.exp(-lr_t * g_normed * tf.math.sign(var))
        new_var = tf.abs(new_var)

        # We are updating weight here. We don't need to return anything
        var.assign(new_var)

def contains_negative(x):
    return tf.reduce_any(tf.math.less(
        x, 0
    ))

def sin_mapping(x):
    return (tf.math.sin(x * math.pi) + 1) / 2


def identity(x):
    return tf.identity(x)


def leaky_abs_relu(x):
    return tf.abs(tf.nn.leaky_relu(
        x, alpha=0.2
    ))


act_func = leaky_abs_relu


class Network(tf.keras.Model):
    def __init__(self, cnn_layers, dense_layers, dim_h, dim_out, weight_scale=1.0, dropout=0.01):
        super(Network, self).__init__()
        self.weight_scale = weight_scale

        self._layers = []
        self._bias = []
        self._rescale = []
        self._dropouts = []
        for _ in range(cnn_layers):
            self._layers.append(
                tf.keras.layers.Dropout(
                    rate=dropout,
                )
            )
            self._layers[-1]._act_func = None
            self._layers.append(
                tf.keras.layers.Conv2D(
                    64, kernel_size=(3, 3), activation=None,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.01 * weight_scale, maxval=1. * weight_scale),
                    bias_initializer=tf.keras.initializers.RandomUniform(minval=0.01 * weight_scale, maxval=1. * weight_scale),
                )
            )
            self._layers[-1]._act_func = act_func
            self._layers.append(
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            )
            self._layers[-1]._act_func = None

        self._layers.append(
            tf.keras.layers.Flatten()
        )
        self._layers[-1]._act_func = None

        self._layers.append(
            tf.keras.layers.Dropout(
                rate=dropout,
            )
        )
        self._layers[-1]._act_func = None

        for _ in range(dense_layers):
            self._layers.append(
                tf.keras.layers.Dense(
                    units=dim_h, activation=None,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.01 * weight_scale, maxval=1. * weight_scale),
                    bias_initializer=tf.keras.initializers.RandomUniform(minval=0.01 * weight_scale, maxval=1. * weight_scale),
                )
            )
            self._layers[-1]._act_func = act_func

        self._layers.append(
            tf.keras.layers.Dense(
                units=dim_out, activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.01 * weight_scale, maxval=1. * weight_scale),
                bias_initializer=tf.keras.initializers.RandomUniform(minval=0.01 * weight_scale, maxval=1. * weight_scale),
            )
        )
        self._layers[-1]._act_func = act_func

        self._rescale = [None] * len(self._layers)
        self._activation_shift = [None] * len(self._layers)

    #@tf.function
    def __call__(self, x, training=False, *args, **kwargs):
        # Adding any of these makes you unable to run in graph mode!!!
        set_absolute = False
        use_rescaling = False
        use_activation_shift = True

        check_positive_constraint = False

        if check_positive_constraint and contains_negative(x):
            raise RuntimeError("Contains negative numbers.")

        for index, layer in enumerate(self._layers):
            tmp_weights = []
            if set_absolute and len(layer._trainable_weights) > 0:
                assert len(layer._trainable_weights) == 2
                tmp_weights.append(layer._trainable_weights[0])
                layer.kernel = tf.abs(layer._trainable_weights[0])

                tmp_weights.append(layer._trainable_weights[0])
                layer.bias = tf.abs(layer._trainable_weights[0])
            x = layer(x, training=training)

            if check_positive_constraint and contains_negative(x):
                raise RuntimeError("Contains negative numbers.")

            if layer._act_func is not None:
                if use_activation_shift:
                    if self._activation_shift[index] is None:
                        layer_name = layer.name
                        self._activation_shift[index] = tf.Variable(
                            tf.random.uniform(shape=x.shape, minval=0.01 * self.weight_scale, maxval=1.0 * self.weight_scale),
                            name=layer_name + '/activation_shift'
                        )

                    if set_absolute:
                        x -= tf.abs(self._activation_shift[index])
                    else:
                        activation_shift = self._activation_shift[index]
                        x -= activation_shift
                else:
                    x -= 0.5

                x = layer._act_func(x)

            if use_rescaling:
                if self._rescale[index] is None:
                    self._rescale[index] = tf.Variable(
                        tf.random.uniform(shape=x.shape, minval=0.01, maxval=1.)
                    )

                if set_absolute:
                    x *= tf.abs(self._rescale[index])
                else:
                    x *= self._rescale[index]

            if check_positive_constraint and contains_negative(x):
                raise RuntimeError("Contains negative numbers.")

            if set_absolute and len(tmp_weights) > 0:
                assert len(tmp_weights) == 2
                layer.kernel = tmp_weights[0]

                layer.bias = tmp_weights[1]

        return x


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(),
])


def main():
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Prepare tensorboard logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_path + '/' + current_time + '_train_search'
    tb_summary_writer = tf.summary.create_file_writer(log_dir)

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the training dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # for mnist add dim
    # x_train = tf.cast(tf.expand_dims(x_train, axis=-1), dtype=tf.float32) / 255
    # x_test = tf.cast(tf.expand_dims(x_test, axis=-1), dtype=tf.float32) / 255

    x_train = tf.cast(x_train, dtype=tf.float32) / 255
    x_test = tf.cast(x_test, dtype=tf.float32) / 255

    x_test_size = x_test.shape[0]

    x_all = tf.concat([x_train, x_test], axis=0)
    x_std = tf.math.reduce_std(x_all, axis=0, keepdims=True)
    x_mean = tf.math.reduce_mean(x_all, axis=0, keepdims=True)
    x_all = tf.math.divide_no_nan((x_all - x_mean), tf.math.sqrt(x_std))

    # make positive
    min_all = tf.reduce_min(x_all)
    max_all = tf.reduce_max(x_all)
    x_all = -0.5 + (x_all - min_all) * (0.5 - (-0.5)) / (max_all - min_all)
    x_all += 0.5

    x_train = x_all[:-x_test_size]
    x_test = x_all[-x_test_size:]

    val_size = int(x_train.shape[0] * args.val_portion)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(10000, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.shuffle(10000, reshuffle_each_iteration=True)
    val_dataset = val_dataset.batch(args.batch_size, drop_remainder=True)
    val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)

    # Prepare the validation dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(10000, reshuffle_each_iteration=True)
    test_dataset = test_dataset.batch(args.batch_size, drop_remainder=True)
    test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

    model = Network(cnn_layers=3, dense_layers=3, dim_h=2000, dim_out=10, weight_scale=0.1)

    scheduler = COSINEANNEALINGLR(
        initial_learning_rate=args.learning_rate, decay_steps=float(args.epochs), min_lr=args.learning_rate_min
    )
    # optimizer = tf.optimizers.SGDW(
    #    learning_rate=scheduler.get_lr(),
    #    momentum=args.momentum,
    #    weight_decay=args.weight_decay
    # )

    #optimizer = tf.keras.optimizers.Adam(
    #    learning_rate=0.001,
    #    beta_1=0.9,
    #    beta_2=0.999,
    #    epsilon=1e-07,
    #    amsgrad=False,
    #)

    optimizer = Madam(bind_lr=True, learning_rate=0.5)

    patience = 40

    best_obj = float('inf')
    best_test_acc = float('-inf')
    patience_left = patience

    epoch = 0
    while True:
        scheduler.step()
        lr = scheduler.get_lr()
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj = train(train_dataset, model, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)
        logging.info('train_obj %f', train_obj)

        if train_obj < best_obj:
            best_obj = train_obj
            patience_left = patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            optimizer.lr.assign(optimizer.lr / 10)
            tf.print("LR reduced!------------------------------------------------------------")
            patience_left = patience
            best_obj = train_obj

        # validation
        test_acc, test_obj = infer(test_dataset, model, criterion)
        logging.info('test_acc %f', test_acc)

        if best_test_acc < test_acc:
            best_test_acc = test_acc

        logging.info('(best train_obj %f)', best_obj)
        logging.info('(best test_acc %f)', best_test_acc)

        # with tb_summary_writer.as_default():
        # tf.summary.scalar('train_obj', train_obj, step=epoch)
        # tf.summary.scalar('train_acc', train_acc, step=epoch)
        # tf.summary.scalar('valid_obj', valid_obj, step=epoch)
        # tf.summary.scalar('valid_acc', valid_acc, step=epoch)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))

        epoch += 1


def train(train_dataset, model, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    step = -1
    for input, target in train_dataset:
        step += 1
        n = input.shape[0]

        with tf.GradientTape() as tape:
            logits = model(input, training=True)
            loss = criterion(y_true=target, y_pred=logits)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.numpy(), n)
        top1.update(prec1.numpy(), n)
        top5.update(prec5.numpy(), n)

        # if step % args.report_freq == 0:
        #    logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(valid_queue):
        logits = model(input, training=False)
        loss = criterion(y_true=target, y_pred=logits)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.shape[0]
        objs.update(loss.numpy(), n)
        top1.update(prec1.numpy(), n)
        top5.update(prec5.numpy(), n)

        # if step % args.report_freq == 0:
        #    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
