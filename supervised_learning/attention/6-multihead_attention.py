#!/usr/bin/env python3
"""Multi Head Attention Module"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """"""

    def __init__(self, dm, h):
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def reshape_tensor(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, Q, K, V, mask):
        batch = Q.shape[0]

        V = self.Wv(V)
        Q = self.Wq(Q)
        K = self.Wk(K)

        V = self.reshape_tensor(V, batch)
        Q = self.reshape_tensor(Q, batch)
        K = self.reshape_tensor(K, batch)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch, -1, self.dm))
        output = self.linear(concat_attention)

        return output, weights
