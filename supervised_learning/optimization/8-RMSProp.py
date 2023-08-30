#!/usr/bin/env python3
"""NN with RMSProp"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm"""

    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    train = optimizer.minimize(loss)
    return train
