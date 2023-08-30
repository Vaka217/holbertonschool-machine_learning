#!/usr/bin/env python3
"""Batch Norm Layer with tf"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network in
    tensorflow"""
    heetal = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=heetal)
    Z = layer(prev)

    beta = tf.zeros(shape=[n], name="beta")
    gamma = tf.ones(shape=[n], name="gamma")

    m, v = tf.nn.moments(Z)

    batch_norm = tf.nn.batch_normalization(prev, m, v, beta, gamma, 1e-8)

    if activation is None:
        return batch_norm

    return activation(batch_norm)
