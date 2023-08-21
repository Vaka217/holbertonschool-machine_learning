#!/usr/bin/env python3
"""Layers"""


import tensorflow as tf

def create_layer(prev, n, activation):
	"""Creates a layer of the NN"""
	layer = tf.layers.Dense(n, activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name="layer")
	return layer(prev)

