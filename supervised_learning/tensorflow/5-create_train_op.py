#!/usr/bin/env python3
"""Training"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
