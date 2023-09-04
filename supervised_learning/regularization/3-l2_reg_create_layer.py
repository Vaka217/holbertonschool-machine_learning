#!/usr/bin/env python3
"""creates a tensorflow layer that includes L2 regularization"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization

    Arguments:
        prev -- previous layer
        n -- number of nodes
        activation -- activation function
        lambtha -- regularization parameter

    Returns:
        layer -- tensorflow layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))

    return tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                           kernel_regularizer=tf.keras.regularizers.L2(
                               lambtha))(prev)
