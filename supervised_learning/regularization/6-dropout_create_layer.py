#!/usr/bin/env python3
"""creates a tensorflow layer that includes Dropout regularization"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a tensorflow layer that includes Dropout regularization

    Arguments:
        prev -- previous layer
        n -- number of nodes
        activation -- activation function
        keep_prob -- probability that a node will be kept

    Returns:
        layer -- tensorflow layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))

    dropping = tf.layers.Dropout(keep_prob)

    return tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                           kernel_regularizer=dropping)(prev)
