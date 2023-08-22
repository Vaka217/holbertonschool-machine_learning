#!/usr/bin/env python3
"""Layers"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer of the NN"""
    heetal = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=heetal, name="layer")
    return layer(prev)
