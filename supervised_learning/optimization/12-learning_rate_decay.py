#!/usr/bin/env python3
"""Learning Rate Decay"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay using tensorflow"""
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate)
