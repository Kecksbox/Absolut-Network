import time

import tensorflow as tf
import numpy as np

from Helpers.hessian_vector_product import _forward_over_back_hvp


def reshape_flattend(params_1d, shapes, part, n_tensors):
    params = tf.dynamic_partition(params_1d, part, n_tensors)
    result = []
    for i, (shape, param) in enumerate(zip(shapes, params)):
        result.append(tf.reshape(param, shape))
    return result


def get_natural_gradient(model, loss_fun, train_x, train_y):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss = loss_fun(model, train_x, train_y)

        inner_tape_grads = inner_tape.gradient(loss, model.trainable_variables)
        g_vec = tf.expand_dims(tf.dynamic_stitch(idx, inner_tape_grads), axis=-1)

    h = outer_tape.jacobian(g_vec, model.trainable_variables)

    for i in range(len(h)):
        h[i] = tf.reshape(h[i], shape=(h[i].shape[0], -1))
    h_mat = tf.concat(h, axis=-1)

    eps = 1e-3
    eye_eps = tf.eye(h_mat.shape[0]) * eps
    update = tf.linalg.solve((h_mat + eye_eps), g_vec)

    return reshape_flattend(tf.squeeze(update), shapes, part, n_tensors)


def get_natural_gradient_scale(model, loss_fun, train_x, train_y):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss = loss_fun(model, train_x, train_y)

        inner_tape_grads = inner_tape.gradient(loss, model.trainable_variables)
        g_vec = tf.expand_dims(tf.dynamic_stitch(idx, inner_tape_grads), axis=-1)

    h = outer_tape.jacobian(g_vec, model.trainable_variables)

    for i in range(len(h)):
        h[i] = tf.reshape(h[i], shape=(h[i].shape[0], -1))
    h_mat = tf.concat(h, axis=-1)

    eps = 1e-3
    eye_eps = tf.eye(h_mat.shape[0]) * eps
    scale = tf.linalg.inv(tf.math.abs(h_mat + eye_eps))
    update = tf.matmul(scale, g_vec)
    # H*a*g = g <=> g*a = H^-1 * g <=>

    return reshape_flattend(tf.squeeze(update), shapes, part, n_tensors)


@tf.function(experimental_relax_shapes=True)
def inner_evaluation(z, r, d, r_k_dot, vector):
    a = r_k_dot / tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(d, axis=-1)), tf.expand_dims(z, axis=-1)))
    vector += a * d
    r_old = r
    r -= a * z

    r_k_new_dot = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(r, axis=-1)), tf.expand_dims(r, axis=-1)))
    polak_ribiere = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(r, axis=-1)), tf.expand_dims(r - r_old, axis=-1)))
    beta = polak_ribiere / r_k_dot
    d = r + beta * d

    r_norm = tf.norm(
        r, ord='euclidean'
    )

    return r, d, vector, r_k_new_dot, r_norm


def get_natural_gradient_hessian_free_back_over_back(model, loss_fun, train_x, train_y, max_iterations):
    start = time.time()

    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss = loss_fun(model, train_x, train_y)
        inner_tape_grads = inner_tape.gradient(loss, model.trainable_variables)
    outer_tape.stop_recording()

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    delta_model = tf.dynamic_stitch(idx, inner_tape_grads)
    inital_guess = -delta_model
    vector = inital_guess

    b = -delta_model

    hvp = outer_tape.gradient(
        inner_tape_grads, model.trainable_variables, output_gradients=reshape_flattend(vector, shapes, part, n_tensors)
    )
    r = b - tf.dynamic_stitch(idx, hvp)
    d = r

    r_k_dot = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(r, axis=-1)), tf.expand_dims(r, axis=-1)))

    inital_r_norm = tf.norm(
        r, ord='euclidean'
    )
    # tf.print(inital_r_norm)

    optimium = [
        inital_r_norm,
        vector,
    ]
    for _ in range(max_iterations):
        hdp = outer_tape.gradient(
            inner_tape_grads, model.trainable_variables, output_gradients=reshape_flattend(d, shapes, part, n_tensors)
        )

        z = tf.dynamic_stitch(idx, hdp)

        r, d, vector, r_k_dot, r_norm = inner_evaluation(z, r, d, r_k_dot, vector)

        # tf.print(r_norm)
        if r_norm < optimium[0]:
            optimium[0] = r_norm
            optimium[1] = vector

    end = time.time()
    tf.print(f'Gradloop took {end - start}')
    tf.print(f'back over back optimium was {optimium[0]}')

    natural_gradient = -optimium[1]

    update = natural_gradient

    return reshape_flattend(update, shapes, part, n_tensors)


def get_natural_gradient_hessian_free_forward_over_back(model, loss_fun, train_x, train_y, max_iterations):
    start = time.time()

    with tf.GradientTape() as inner_tape:
        loss = loss_fun(model, train_x, train_y)
    inner_tape_grads = inner_tape.gradient(loss, model.trainable_variables)

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    delta_model = tf.dynamic_stitch(idx, inner_tape_grads)
    inital_guess = -delta_model
    vector = inital_guess

    b = delta_model

    hvp = _forward_over_back_hvp(
        model, loss_fun, train_x, train_y, reshape_flattend(vector, shapes, part, n_tensors)
    )
    r = b - tf.dynamic_stitch(idx, hvp)
    d = r

    r_k_dot = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(r, axis=-1)), tf.expand_dims(r, axis=-1)))

    inital_r_norm = tf.norm(
        r, ord='euclidean'
    )
    tf.print(inital_r_norm)

    optimium = [
        inital_r_norm,
        vector,
    ]
    for _ in range(max_iterations):
        hdp = _forward_over_back_hvp(
            model, loss_fun, train_x, train_y, reshape_flattend(d, shapes, part, n_tensors)
        )

        z = tf.dynamic_stitch(idx, hdp)

        r, d, vector, r_k_dot, r_norm = inner_evaluation(z, r, d, r_k_dot, vector)

        tf.print(r_norm)
        if r_norm < optimium[0]:
            optimium[0] = r_norm
            optimium[1] = vector

    end = time.time()
    tf.print(f'Gradloop took {end - start}')
    tf.print(f'front over back optimium was {optimium[0]}')

    return reshape_flattend(-optimium[1], shapes, part, n_tensors)


def get_natural_gradient_hessian_free_jacobian_preconditioned(model, loss_fun, train_x, train_y, max_iterations):
    start = time.time()

    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape(persistent=True) as inner_tape:
            loss = loss_fun(model, train_x, train_y)
        inner_tape_grads = inner_tape.gradient(loss, model.trainable_variables)
    outer_tape.stop_recording()
    inner_tape_jacobian = inner_tape.jacobian(loss, model.trainable_variables)

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    delta_model = tf.dynamic_stitch(idx, inner_tape_grads)
    jacobian = tf.linalg.diag(tf.dynamic_stitch(idx, inner_tape_jacobian))

    inital_guess = -delta_model
    vector = inital_guess

    b = delta_model

    C = tf.matmul(jacobian, tf.transpose(jacobian))

    hvp = outer_tape.gradient(
        inner_tape_grads, model.trainable_variables, output_gradients=reshape_flattend(vector, shapes, part, n_tensors)
    )
    r = b - tf.dynamic_stitch(idx, hvp)
    h = tf.squeeze(tf.matmul(C, tf.expand_dims(r, axis=-1)))
    d = h

    r_h_dot = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(r, axis=-1)), tf.expand_dims(h, axis=-1)))

    inital_r_norm = tf.norm(
        r, ord='euclidean'
    )
    # tf.print(inital_r_norm)

    optimium = [
        inital_r_norm,
        vector,
    ]
    for _ in range(max_iterations):
        hdp = outer_tape.gradient(
            inner_tape_grads, model.trainable_variables, output_gradients=reshape_flattend(d, shapes, part, n_tensors)
        )

        z = tf.dynamic_stitch(idx, hdp)

        a = r_h_dot / tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(d, axis=-1)), tf.expand_dims(z, axis=-1)))
        vector += a * d
        r -= a * z
        h = tf.squeeze(tf.matmul(C, tf.expand_dims(r, axis=-1)))

        r_h_new_dot = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(r, axis=-1)), tf.expand_dims(h, axis=-1)))
        beta = r_h_new_dot / r_h_dot
        d = h + beta * d

        r_h_dot = r_h_new_dot

        r_norm = tf.norm(
            r, ord='euclidean'
        )

        tf.print(r_norm)
        if r_norm < optimium[0]:
            optimium[0] = r_norm
            optimium[1] = vector

    end = time.time()
    tf.print(f'Gradloop took {end - start}')

    return reshape_flattend(-optimium[1], shapes, part, n_tensors)


def get_natural_gradient_hessian_free_jacobian_preconditioned_v2(model, loss_fun, train_x, train_y, max_iterations):
    start = time.time()

    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape(persistent=True) as inner_tape:
            loss = loss_fun(model, train_x, train_y)
        inner_tape_grads = inner_tape.gradient(loss, model.trainable_variables)
    outer_tape.stop_recording()
    inner_tape_jacobian = inner_tape.jacobian(loss, model.trainable_variables)

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    delta_model = tf.dynamic_stitch(idx, inner_tape_grads)
    jacobian = tf.dynamic_stitch(idx, inner_tape_jacobian)
    inital_guess = -delta_model
    vector = inital_guess

    b = delta_model

    Jvp = jacobian * vector
    hJvpp = _forward_over_back_hvp(
        model, loss_fun, train_x, train_y, reshape_flattend(Jvp, shapes, part, n_tensors)
    )
    JHvpP = tf.squeeze(
        tf.matmul(tf.transpose(tf.expand_dims(jacobian, axis=-1)),
                  tf.expand_dims(tf.dynamic_stitch(idx, hJvpp), axis=-1))
    )
    r = b - JHvpP
    d = r

    r_k_dot = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(r, axis=-1)), tf.expand_dims(r, axis=-1)))

    inital_r_norm = tf.norm(
        r, ord='euclidean'
    )
    tf.print(inital_r_norm)

    optimium = [
        inital_r_norm,
        vector,
    ]
    for _ in range(max_iterations):
        Jdp = jacobian * d
        hJdpp = _forward_over_back_hvp(
            model, loss_fun, train_x, train_y, reshape_flattend(Jdp, shapes, part, n_tensors)
        )
        JHdpP = tf.squeeze(
            tf.matmul(tf.transpose(tf.expand_dims(jacobian, axis=-1)),
                      tf.expand_dims(tf.dynamic_stitch(idx, hJdpp), axis=-1))
        )
        hdp = _forward_over_back_hvp(
            model, loss_fun, train_x, train_y, reshape_flattend(d, shapes, part, n_tensors)
        )

        z = tf.dynamic_stitch(idx, hdp)

        r, d, vector, r_k_dot, r_norm = inner_evaluation(z, r, d, r_k_dot, vector)

        tf.print(r_norm)
        if r_norm < optimium[0]:
            optimium[0] = r_norm
            optimium[1] = vector

    end = time.time()
    tf.print(f'Gradloop took {end - start}')
    tf.print(f'front over back optimium was {optimium[0]}')

    return reshape_flattend(-optimium[1], shapes, part, n_tensors)
