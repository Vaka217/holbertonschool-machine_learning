#!/usr/bin/env python3
"""Makes a prediction using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network.

    Args:
        network (Model): the network model to test.
        data (np.ndarray): the input data to test the model with.
        labels (np.ndarray): the correct one-hot labels of data.
        verbose (bool, optional): determines if output should be printed during
        the testing process. Defaults to True.

    Returns:
        np.ndarray: the prediction for the data.
    """

    return network.predict(data, verbose=verbose)
