#!/usr/bin/env python3
"""Builds a modified version of the LeNet-5 architecture"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using keras.

    Args:
        X (K.Input): shape (m, 28, 28, 1) containing the input images
        for the network.

    Returns:
        a K.Model compiled to use Adam optimization
        (with default hyperparameters) and accuracy metrics.
    """

    model = K.Sequential()

    model._set_inputs(X)

    model.add(K.layers.Conv2D(6, 5, padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.MaxPool2D(2, 2))

    model.add(K.layers.Conv2D(16, 5, padding='valid',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.MaxPool2D(2, 2))

    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(120, K.activations.relu,
                             kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.Dense(84, K.activations.relu,
                             kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.Dense(10, K.activations.softmax,
                             kernel_initializer=K.initializers.HeNormal))

    adam = K.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=adam,
                  metrics=['accuracy'])

    return model
