#!/usr/bin/env python3
"""Identity Block Module"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds an projection block.

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3, F12, respectively:
            F11 is the number of filters in the first 1x1 convolution.
            F3 is the number of filters in the 3x3 convolution.
            F12 is the number of filters in the second 1x1 convolution as
            well as the 1x1 convolution in the shortcut connection.
        s: the stride of the first convolution in both the main path and
        the shortcut connection.

    Returns:
        The activated output of the projection block.
    """

    initializer = K.initializers.he_normal

    conv1_0 = K.layers.Conv2D(
        filters[0], 1, kernel_initializer=initializer, strides=s)(A_prev)
    batch_norm1_0 = K.layers.BatchNormalization()(conv1_0)
    relu1_0 = K.layers.Activation(K.activations.relu)(batch_norm1_0)

    conv3 = K.layers.Conv2D(
        filters[1], 3, kernel_initializer=initializer, padding="same")(relu1_0)
    batch_norm3 = K.layers.BatchNormalization()(conv3)
    relu3 = K.layers.Activation(K.activations.relu)(batch_norm3)

    conv1_1 = K.layers.Conv2D(
        filters[2], 1, kernel_initializer=initializer)(relu3)
    conv1_2 = K.layers.Conv2D(
        filters[2], 1, kernel_initializer=initializer, strides=s)(A_prev)
    batch_norm1_1 = K.layers.BatchNormalization()(conv1_1)
    batch_norm1_2 = K.layers.BatchNormalization()(conv1_2)
    id_block = K.layers.Add()([batch_norm1_1, batch_norm1_2])

    return K.layers.Activation(K.activations.relu)(id_block)
