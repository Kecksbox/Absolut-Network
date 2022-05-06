import tensorflow as tf

from tensorflow.python.eager import forwardprop


def _forward_over_back_hvp(model, loss_func, train_x, train_y, vector):
    with forwardprop.ForwardAccumulator(
            model.trainable_variables, vector) as acc:
        with tf.GradientTape() as grad_tape:
            loss = loss_func(model, train_x, train_y)
        grads = grad_tape.gradient(loss, model.trainable_variables)
    return acc.jvp(grads)


def _back_over_forward_hvp(model, loss_func, train_x, train_y, vector):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(model.trainable_variables)
        with forwardprop.ForwardAccumulator(
                model.trainable_variables, vector) as acc:
            loss = loss_func(model, train_x, train_y)
    return grad_tape.gradient(acc.jvp(loss), model.trainable_variables)


def _tf_gradients_forward_over_back_hvp(model, loss_func, train_x, train_y, vector):
    with tf.GradientTape() as grad_tape:
        loss = loss_func(model, train_x, train_y)
    variables = model.trainable_variables
    grads = grad_tape.gradient(loss, variables)
    helpers = tf.nest.map_structure(tf.ones_like, grads)
    transposing = tf.gradients(grads, variables, helpers)
    return tf.gradients(transposing, helpers, vector)


def _back_over_back_hvp(model, loss_func, train_x, train_y, vector):
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss = loss_func(model, train_x, train_y)
        grads = inner_tape.gradient(loss, model.trainable_variables)
    return outer_tape.gradient(
        grads, model.trainable_variables, output_gradients=vector)
