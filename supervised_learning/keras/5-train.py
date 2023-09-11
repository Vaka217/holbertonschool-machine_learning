#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and analyze validation
    data

    Args:
        network (Model): the model to train
        data (numpy.ndarray): the input data
        labels (numpy.ndarray): the labels of data
        batch_size (int): the size of the batch used for mini-batch gradient
        descent
        epochs (int): the number of passes through data for mini-batch
        gradient descent
        validation_data (tuple): the data to validate the model.
        Defaults to None
        verbose (bool, optional): boolean that determines if output should be
        printed during training. Defaults to True.
        shuffle (bool, optional): boolean that determines whether to shuffle
        the batches every epoch. Normally, it is a good idea to shuffle, but
        for reproducibility, we have chosen to set the default to False.

    Returns:
        History: History object generated after training the model
    """

    return network.fit(data, labels, batch_size, epochs, verbose,
                       shuffle=shuffle, validation_data=validation_data)
