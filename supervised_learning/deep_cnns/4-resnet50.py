#!/usr/bin/env python3
"""ResNet-50 Module"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds an ResNet-50

    Returns:
        The keras model.
    """

    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal

    conv2d = K.layers.Conv2D(
        64, 7, kernel_initializer=initializer, strides=2, padding="same")(X)
    batch_norm = K.layers.BatchNormalization()(conv2d)
    relu = K.layers.Activation(K.activations.relu)(batch_norm)
    max_pool = K.layers.MaxPool2D(3, 2, padding="same")(relu)

    id_block_0 = projection_block(max_pool, [64, 64, 256], s=1)
    id_block_1 = identity_block(id_block_0, [64, 64, 256])
    id_block_2 = identity_block(id_block_1, [64, 64, 256])
    pro_block_0 = projection_block(id_block_2, [128, 128, 512])

    id_block_3 = identity_block(pro_block_0, [128, 128, 512])
    id_block_4 = identity_block(id_block_3, [128, 128, 512])
    id_block_5 = identity_block(id_block_4, [128, 128, 512])
    pro_block_1 = projection_block(id_block_5, [256, 256, 1024])

    id_block_6 = identity_block(pro_block_1, [256, 256, 1024])
    id_block_7 = identity_block(id_block_6, [256, 256, 1024])
    id_block_8 = identity_block(id_block_7, [256, 256, 1024])
    id_block_9 = identity_block(id_block_8, [256, 256, 1024])
    id_block_10 = identity_block(id_block_9, [256, 256, 1024])
    pro_block_2 = projection_block(id_block_10, [512, 512, 2048])

    id_block_11 = identity_block(pro_block_2, [512, 512, 2048])
    id_block_12 = identity_block(id_block_11, [512, 512, 2048])

    average_pool = K.layers.AveragePooling2D(7, 1)(id_block_12)
    Y = K.layers.Dense(
        1000, "softmax", kernel_initializer=initializer)(average_pool)

    return K.Model(inputs=X, outputs=Y)
