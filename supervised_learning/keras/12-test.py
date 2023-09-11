#!/usr/bin/env python3
"""Tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network.

    Args:
        network (Model): the network model to test.
        data (np.ndarray): the input data to test the model with.
        labels (np.ndarray): the correct one-hot labels of data.
        verbose (bool, optional): determines if output should be printed during
        the testing process. Defaults to True.

    Returns:
        List: the loss and accuracy of the model with the testing data,
        respectively.
    """

    return network.evaluate(data, labels, verbose=1)
