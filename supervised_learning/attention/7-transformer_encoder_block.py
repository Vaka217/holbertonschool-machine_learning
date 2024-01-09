#!/usr/bin/env python3
"""Transformer Encoder Block Module"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def __call__(self, x, training, mask=None):
        multihead_output = self.mha(x, x, x, mask)

        multihead_output = self.dropout1(multihead_output, training=training)

        addnorm_output = self.layernorm1(multihead_output + x)

        dense_output = self.dense_hidden(addnorm_output)
        dense_output = self.dense_output(dense_output)

        dense_output = self.dropout2(dense_output, training=training)

        encoder_output = self.layernorm2(dense_output + addnorm_output)

        return encoder_output
