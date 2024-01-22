#!/usr/bin/env python3
"""Transformer Encoder Block Module"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Create an encoder block for a transformer:

    Class constructor def __init__(self, dm, h, hidden, drop_rate=0.1):
    dm - the dimensionality of the model
    h - the number of heads
    hidden - the number of hidden units in the fully connected layer
    drop_rate - the dropout rate
    Sets the following public instance attributes:
    mha - a MultiHeadAttention layer
    dense_hidden - the hidden dense layer with hidden units and relu activation
    dense_output - the output dense layer with dm units
    layernorm1 - the first layer norm layer, with epsilon=1e-6
    layernorm2 - the second layer norm layer, with epsilon=1e-6
    dropout1 - the first dropout layer
    dropout2 - the second dropout layer
    Public instance method call(self, x, training, mask=None)"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def __call__(self, x, training, mask=None):
        multihead_output, _ = self.mha(x, x, x, mask)

        multihead_output = self.dropout1(multihead_output, training=training)

        addnorm_output = self.layernorm1(x + multihead_output)

        dense_output = self.dense_hidden(addnorm_output)
        dense_output = self.dense_output(dense_output)

        dense_output = self.dropout2(dense_output, training=training)

        encoder_output = self.layernorm2(addnorm_output + dense_output)

        return encoder_output
