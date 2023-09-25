#!/usr/bin/env python3
"""Inception Block Module"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block

    Args:
        A_prev (K.Input): the output from the previous layer.

        filters (tuple/list): contains F1, F3R, F3,F5R, F5, FPP, respectively:
            F1 is the number of filters in the 1x1 convolution.
            F3R is the number of filters in the 1x1 convolution before the 3x3
            convolution.
            F3 is the number of filters in the 3x3 convolution.
            F5R is the number of filters in the 1x1 convolution before the 5x5
            convolution.
            F5 is the number of filters in the 5x5 convolution.
            FPP is the number of filters in the 1x1 convolution after the max
            pooling.

    Returns:
        The concatenated output of the inception block.
    """

    conv1_0 = K.layers.Conv2D(filters[0], 1,
                              activation=K.activations.relu)(A_prev)

    conv1_1 = K.layers.Conv2D(filters[1], 1,
                              activation=K.activations.relu)(A_prev)
    conv3 = K.layers.Conv2D(filters[2], 3,
                            activation=K.activations.relu,
                            padding="same")(conv1_1)

    conv1_2 = K.layers.Conv2D(filters[3], 1,
                              activation=K.activations.relu)(A_prev)
    conv5 = K.layers.Conv2D(filters[4], 5,
                            activation=K.activations.relu,
                            padding="same")(conv1_2)

    max_pool = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    conv1_3 = K.layers.Conv2D(filters[5], 1,
                              activation=K.activations.relu)(max_pool)

    return K.layers.Concatenate()([conv1_0, conv3, conv5, conv1_3])
