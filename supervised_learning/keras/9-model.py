#!/usr/bin/env python3
"""Save and Load Model"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model.

    Args:
        network (Model): the model to save.
        filename (str): the path of the file that the model should be saved to.
    """

    network.save(filename)


def load_model(filename):
    """Loads an entire model.

    Args:
        filename (str): the path of the file that the model should be loaded
        from.

    Returns:
        Model: the loaded model.
    """

    return K.models.load_model(filename)
