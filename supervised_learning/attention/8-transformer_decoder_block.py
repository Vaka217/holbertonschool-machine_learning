#!/usr/bin/env python3
"""Transformer Encoder Block Module"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Create a decoder block for a transformer:

    Class constructor def __init__(self, dm, h, hidden, drop_rate=0.1):
    dm - the dimensionality of the model
    h - the number of heads
    hidden - the number of hidden units in the fully connected layer
    drop_rate - the dropout rate
    Sets the following public instance attributes:
    mha1 - the first MultiHeadAttention layer
    mha2 - the second MultiHeadAttention layer
    dense_hidden - the hidden dense layer with hidden units and relu activation
    dense_output - the output dense layer with dm units
    layernorm1 - the first layer norm layer, with epsilon=1e-6
    layernorm2 - the second layer norm layer, with epsilon=1e-6
    layernorm3 - the third layer norm layer, with epsilon=1e-6
    dropout1 - the first dropout layer
    dropout2 - the second dropout layer
    dropout3 - the third dropout layer

    Public instance method def call(self, x, encoder_output, training,
    look_ahead_mask, padding_mask)"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """x - a tensor of shape (batch, target_seq_len, dm)containing the
        input to the decoder block
        encoder_output - a tensor of shape (batch, input_seq_len, dm)containing
        the output of the encoder
        training - a boolean to determine if the model is training
        look_ahead_mask - the mask to be applied to the first multi head
        attention layer
        padding_mask - the mask to be applied to the second multi head
        attention layer
        Returns: a tensor of shape (batch, target_seq_len, dm) containing
        the block’s output"""
        attention1, weights1 = self.mha1(x, x, x, look_ahead_mask)

        attention1 = self.dropout1(attention1, training=training)

        addnorm_output1 = self.layernorm1(x + attention1)

        attention2, weights2 = self.mha2(addnorm_output1, encoder_output,
                                         encoder_output, padding_mask)

        attention2 = self.dropout2(attention2, training=training)

        addnorm_output2 = self.layernorm2(addnorm_output1 + attention2)

        hidden_output = self.dense_hidden(addnorm_output2)

        dense_output = self.dense_output(hidden_output)

        dense_output = self.dropout3(dense_output, training=training)

        encoder_output = self.layernorm3(dense_output + addnorm_output2)

        return encoder_output
