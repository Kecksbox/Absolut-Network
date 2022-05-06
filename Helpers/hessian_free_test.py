import tensorflow as tf

from Helpers.hessianFreeOptimizer import get_natural_gradient_hessian_free_back_over_back, get_natural_gradient, \
    get_natural_gradient_hessian_free_jacobian_preconditioned, \
    get_natural_gradient_hessian_free_jacobian_preconditioned_v2, get_natural_gradient_scale

model = tf.keras.Sequential([
    #tf.keras.layers.Dense(69, activation='tanh'),
    tf.keras.layers.Dense(69, activation='sigmoid'),
    tf.keras.layers.Dense(69, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])

data_x = tf.constant([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
], dtype=tf.float32)

data_y = tf.constant([
    [0],
    [1],
    [1],
    [0],
], dtype=tf.float32)

counter = 0

model(data_x)


def loss_function(model, data_x, data_y):
    pred = tf.math.softplus(model(data_x))
    loss = tf.reduce_mean(
        tf.math.square(pred - data_y)
    )
    global counter
    counter += 1
    tf.print(counter)
    tf.print(loss)
    return loss


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


def test_function():
    while True:
        grads = get_natural_gradient_hessian_free_back_over_back(
            model,
            loss_function,
            data_x,
            data_y,
            max_iterations=50,
        )
        """
        60: 0.111990571
        200: 0.013083607
        with tf.GradientTape() as inner_tape:
            loss = loss_function(model, data_x, data_y)
        grads = inner_tape.gradient(loss, model.trainable_variables)
        """

        optimizer.apply_gradients(zip(grads, model.trainable_variables))


test_function()
