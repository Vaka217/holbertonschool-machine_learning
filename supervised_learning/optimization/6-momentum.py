#!/usr/bin/env python3
"""NN with Gradient Descent with momentum"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm"""

    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train = optimizer.minimize(loss)
    return train
