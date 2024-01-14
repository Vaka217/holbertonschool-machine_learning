#!/usr/bin/env python3
"""Multi Head Attention Module"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi head attention

    Class constructor def __init__(self, dm, h):
        dm is an integer representing the dimensionality of the model
        h is an integer representing the number of heads
        dm is divisible by h
        Sets the following public instance attributes:
        h - the number of heads
        dm - the dimensionality of the model
        depth - the depth of each attention head
        Wq - a Dense layer with dm units, used to generate the query matrix
        Wk - a Dense layer with dm units, used to generate the key matrix
        Wv - a Dense layer with dm units, used to generate the value matrix
        linear - a Dense layer with dm units, used to generate the attention
        output

    Public instance method def call(self, Q, K, V, mask)"""

    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def reshape_tensor(self, x, batch):
        """Split over the last axis of any given x"""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, Q, K, V, mask):
        """Q is a tensor of shape (batch, seq_len_q, dk) containing the input
        to generate the query matrix
        K is a tensor of shape (batch, seq_len_v, dk) containing the input to
        generate the key matrix
        V is a tensor of shape (batch, seq_len_v, dv) containing the input to
        generate the value matrix
        mask is always None

        Returns: output, weights
        outputa tensor with its last two dimensions as (..., seq_len_q, dm)
        containing the scaled dot product attention
        weights a tensor with its last three dimensions as
        (..., h, seq_len_q, seq_len_v) containing the attention weights"""
        batch = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.reshape_tensor(Q, batch)
        K = self.reshape_tensor(K, batch)
        V = self.reshape_tensor(V, batch)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch, -1, self.dm))
        output = self.linear(concat_attention)

        return output, weights
