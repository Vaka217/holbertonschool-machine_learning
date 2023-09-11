#!/usr/bin/env python3
"""Adam optimization for a keras model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics

    Args:
        network (Model): model to optimize
        alpha (float): learning rate
        beta1 (float): first Adam optimization parameter
        beta2 (float): second Adam optimization parameter
    """

    adam = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss="categorical_crossentropy", optimizer=adam)
