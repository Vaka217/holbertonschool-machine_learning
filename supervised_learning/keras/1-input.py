#!/usr/bin/env python3
"""Builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library

    Args:
        nx (int): number of input features to the network
        layers (list): list containing the number of nodes in each layer of
        the network
        activations (list): list containing the activation functions used for
        each layer of the network
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout

    Returns:
        model: the keras model
    """

    outputs = inputs = K.layers.Input(shape=(nx,))
    for i in range(len(layers)):
        outputs = K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=K.regularizers.L2(
                                     lambtha))(outputs)
        if i != len(layers) - 1:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)

    model = K.Model(inputs, outputs)
    return model
