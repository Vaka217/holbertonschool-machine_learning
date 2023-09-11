#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent, analyze validation
    data, uses early stopping, learning rate decay and save the best iteration
    of the model.

    Args:
        network (Model): the model to train.
        data (numpy.ndarray): the input data.
        labels (numpy.ndarray): the labels of data.
        batch_size (int): the size of the batch used for mini-batch gradient
        descent.
        epochs (int): the number of passes through data for mini-batch
        gradient descent.
        validation_data (tuple, optional): the data to validate the model.
        Defaults to None.
        early_stopping (bool, optional): indicates whether early stopping
        should be used. Defaults to False.
        patience (int, optional): the patience used for early stopping.
        Defaults to 0.
        learning_rate_decay (bool, optional): indicates whether learning rate
        decay should be used. Defaults to False.
        alpha (float, optional): the initial learning rate. Defaults to 0.1.
        decay_rate (int, optional): the decay rate. Defaults to 1.
        save_best (bool, optional): indicates whether to save the model after
        each epoch if it is the best. Defaults to False.
        filepath (str, optional): the file path where the model should be
        saved. Defaults to None.
        verbose (bool, optional): boolean that determines if output should be
        printed during training. Defaults to True.
        shuffle (bool, optional): boolean that determines whether to shuffle
        the batches every epoch. Normally, it is a good idea to shuffle, but
        for reproducibility, we have chosen to set the default to False.

    Returns:
        History: History object generated after training the model
    """

    if learning_rate_decay and validation_data:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        learning_rate_decay = K.callbacks.LearningRateScheduler(lr_schedule,
                                                                verbose=True)

    early_stopping = K.callbacks.EarlyStopping(
        patience=patience) if early_stopping and validation_data else None

    save_best = K.callbacks.ModelCheckpoint(
        filepath, save_best_only=True
        ) if save_best and validation_data else None

    callback = [early_stopping, learning_rate_decay, save_best]

    return network.fit(data, labels, batch_size, epochs, verbose,
                       shuffle=shuffle, validation_data=validation_data,
                       callbacks=callback)
