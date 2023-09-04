#!/usr/bin/env python3
"""Calculates the cost of NN with L2 regularization"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """Calculates the cost of NN with L2 regularization

    Arguments:
        cost -- tensor containing cost of NN without L2 regularization

    Returns:
        l2_cost -- tensor containing cost of NN with L2 regularization
    """

    return cost + tf.losses.get_regularization_losses()
