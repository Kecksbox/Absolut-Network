import logging
from typing import Tuple
import tensorflow as tf

from main import Madam, leaky_abs_relu, leaky_abs_relu_up_only

logging.getLogger().setLevel(logging.INFO)


def create_train_batch(batch_size: int, a_bound: Tuple[float, float], b_bound: Tuple[float, float]):
    assert batch_size >= 1
    assert a_bound[1] >= a_bound[0] >= 0
    assert b_bound[1] >= b_bound[0] >= 0

    a = tf.random.uniform(shape=(batch_size,), minval=a_bound[0], maxval=a_bound[1])
    b = tf.random.uniform(shape=(batch_size,), minval=b_bound[0], maxval=a)

    target = a - b
    inp = tf.transpose(tf.stack([a, b]), [1, 0])

    return inp, target


def main():
    batch_size = 200
    a_bound = (0, 1)
    b_bound = a_bound

    optimizer = Madam(bind_lr=False)

    criterion = tf.keras.losses.MeanSquaredError()

    act_func = leaky_abs_relu
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=act_func,
                              kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.0),
                              bias_initializer=tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.0)),
        tf.keras.layers.Dense(20, activation=act_func,
                              kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.0),
                              bias_initializer=tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.0)),
        tf.keras.layers.Dense(1, activation=act_func,
                              kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.0),
                              bias_initializer=tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.0)),
    ])

    epoch = 0

    patience = 20

    best_obj = float('inf')
    patience_left = patience

    while True:
        inp, target = create_train_batch(batch_size, a_bound, b_bound)
        logging.info('epoch %d', epoch)

        # training
        train_obj = 0
        num_runs = 200
        for _ in range(num_runs):
            with tf.GradientTape() as tape:
                logits = model(inp)
                loss = criterion(y_true=target, y_pred=logits)
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_obj += loss
        train_obj /= num_runs
        logging.info('pred %f, true: %f', logits[0], target[0])
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

        epoch += 1


if __name__ == '__main__':
    main()
