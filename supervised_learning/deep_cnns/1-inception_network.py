#!/usr/bin/env python3
"""Inception Network Module"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds an inception network

    Returns:
        The keras model.
    """

    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.HeNormal

    conv2d = K.layers.Conv2D(64, 7, activation=K.activations.relu, strides=2,
                             kernel_initializer=init, padding="same")(X)
    max_pooling2d = K.layers.MaxPooling2D(3, 2, padding="same")(conv2d)
    conv2d_1 = K.layers.Conv2D(64, 1, activation=K.activations.relu,
                               kernel_initializer=init,
                               padding="same")(max_pooling2d)
    conv2d_2 = K.layers.Conv2D(192, 3, activation=K.activations.relu,
                               kernel_initializer=init,
                               padding="same")(conv2d_1)
    max_pooling2d_1 = K.layers.MaxPooling2D(3, 2, padding="same")(conv2d_2)
    concatenate = inception_block(max_pooling2d_1, [64, 96, 128, 16, 32, 32])
    concatenate_1 = inception_block(concatenate, [128, 128, 192, 32, 96, 64])
    max_pooling2d_2 = K.layers.MaxPooling2D(3, 2, padding="same")(concatenate_1)
    concatenate_2 = inception_block(max_pooling2d_2,
                                    [192, 96, 208, 16, 48, 64])
    concatenate_3 = inception_block(concatenate_2, [160, 112, 224, 24, 64, 64])
    concatenate_4 = inception_block(concatenate_3, [128, 128, 256, 24, 64, 64])
    concatenate_5 = inception_block(concatenate_4, [112, 144, 288, 32, 64, 64])
    concatenate_6 = inception_block(concatenate_5,
                                    [256, 160, 320, 32, 128, 128])
    max_pooling2d_3 = K.layers.MaxPooling2D(3, 2, padding="same")(concatenate_6)
    concatenate_7 = inception_block(max_pooling2d_3,
                                    [256, 160, 320, 32, 128, 128])
    concatenate_8 = inception_block(concatenate_7,
                                    [384, 192, 384, 48, 128, 128])
    average_polling2d = K.layers.AveragePooling2D(7, 1)(concatenate_8)
    dropout = K.layers.Dropout(0.4)(average_polling2d)
    softmax = K.layers.Dense(1000, K.activations.softmax,
                             kernel_initializer=init)(dropout)

    return K.Model(inputs=X, outputs=softmax)
