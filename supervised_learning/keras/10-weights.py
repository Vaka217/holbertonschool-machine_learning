#!/usr/bin/env python3
"""Save and Load Weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format="h5"):
    """Saves a model’s weights.

    Args:
        network (Model): the model whose weights should be saved.
        filename (str): the path of the file that the weights should be saved
        to.
        save_format (str, optional): the format in which the weights should be
        saved. Defaults to 'h5'.
    """

    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Loads a model’s weights.

    Args:
        network (Model): the model to which the weights should be loaded.
        filename (str): the path of the file that the weights should be loaded
        from.
    """

    return network.load_weights(filename)
