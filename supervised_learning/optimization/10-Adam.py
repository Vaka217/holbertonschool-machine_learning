#!/usr/bin/env python3
"""NN with Adam"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm"""

    optimizer = tf.train.RMSPropOptimizer(alpha, beta1, beta2, epsilon)
    train = optimizer.minimize(loss)
    return train
