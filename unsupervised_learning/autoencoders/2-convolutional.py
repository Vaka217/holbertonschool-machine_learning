#!/usr/bin/env python3
"""Vanilla Autoencoder Module"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder:

    input_dims is a tuple of integers containing the dimensions of the model
    input
    filters is a list containing the number of filters for each convolutional
    layer in the encoder, respectively
    the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the latent
    space representation
    Each convolution in the encoder should use a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should use a
    filter size of (3, 3) with same padding and relu activation, followed by
    upsampling of size (2, 2)
    The second to last convolution should instead use valid padding
    The last convolution should have the same number of filters as the number
    of channels in input_dims with sigmoid activation and no upsampling

    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    """
    encoder_input = keras.layers.Input(shape=input_dims)
    encoder_output = encoder_input
    for f in filters:
        encoder_output = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(encoder_output)
        encoder_output = keras.layers.MaxPooling2D(
            (2, 2), padding='same')(encoder_output)

    encoder = keras.models.Model(encoder_input, encoder_output)

    decoder_input = keras.layers.Input(shape=latent_dims)
    decoder_output = decoder_input
    for f in reversed(filters[1:]):
        decoder_output = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(decoder_output)
        decoder_output = keras.layers.UpSampling2D((2, 2))(decoder_output)

    decoder_output = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu')(decoder_output)
    decoder_output = keras.layers.UpSampling2D((2, 2))(decoder_output)
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same',
        activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    auto_outputs = encoder(encoder_input)
    auto_outputs = decoder(auto_outputs)
    autoencoder = keras.models.Model(encoder_input, auto_outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
