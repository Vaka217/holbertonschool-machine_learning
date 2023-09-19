#!/usr/bin/env python3
"""Builds a modified version of the LeNet-5 architecture."""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5 architecture with tensorflow.

    Args:
        x (tf.placeholder): shape (m, 28, 28, 1) containing the input images
        for the network.

        y (tf.placeholder): shape (m, 10) containing the one-hot labels for the
        network.

    Returns:
        - a tensor for the softmax activated output.
        - a training operation that utilizes Adam optimization
        (with default hyperparameters).
        - a tensor for the loss of the netowrk.
        - a tensor for the accuracy of the network.
    """
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv_1 = tf.layers.conv2d(x, 6, 5, padding="same", activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2)
    conv_2 = tf.layers.conv2d(max_pool_1, 16, 5, padding="valid",
                              activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2)
    flatten = tf.layers.flatten(max_pool_2)
    full_connected_1 = tf.layers.dense(flatten, 120, activation=tf.nn.relu,
                                       kernel_initializer=he_normal)
    full_connected_2 = tf.layers.dense(full_connected_1, 84,
                                       activation=tf.nn.relu,
                                       kernel_initializer=he_normal)
    y_pred = tf.layers.dense(full_connected_2, 10, activation=tf.math.softmax,
                             kernel_initializer=he_normal)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    pred_labels = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(pred_labels, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return y_pred, train, loss, accuracy
