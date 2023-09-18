#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer of a
neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform backpropagation over a convolutional layer of a neural network.

    Args:
    dZ (numpy.ndarray): Partial derivatives with respect to the unactivated
    output of the convolutional layer.
        Shape: (m, h_new, w_new, c_new)
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output

    A_prev (numpy.ndarray): Output of the previous layer.
        Shape: (m, h_prev, w_prev, c_prev)
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer

    W (numpy.ndarray): Kernels for the convolution.
        Shape: (kh, kw, c_prev, c_new)
        kh is the filter height
        kw is the filter width

    b (numpy.ndarray): Biases applied to the convolution.
        Shape: (1, 1, 1, c_new)

    padding (str): Type of padding used, either 'same' or 'valid'.

    stride (tuple): Strides for the convolution.
        Tuple format: (sh, sw)
        sh: Stride for the height
        sw: Stride for the width

    Returns:
    dA_prev (numpy.ndarray): Partial derivatives with respect to the input of
    the previous layer (A_prev).
        Shape: (m, h_prev, w_prev, c_prev)
    dW (numpy.ndarray): Partial derivatives with respect to the convolutional
    kernels (W).
        Shape: (kh, kw, c_prev, c_new)
    db (numpy.ndarray): Partial derivatives with respect to the biases (b).
        Shape: (1, 1, 1, c_new)
    """

    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    ph, pw = 0, 0

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
        A_prev = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)])

    m, nh, nw, c_new = dZ.shape

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(m):
        a_prev = A_prev[i]
        da_prev = np.zeros(a_prev.shape)

        for h in range(nh):
            for w in range(nw):
                for c in range(c_new):
                    da_prev[h*sh:h*sh+kh, w*sw:w*sw+kw] += W[:, :, :, c] * dZ[
                        i, h, w, c]
                    dW[:, :, :, c] += a_prev[h*sh:h*sh+kh, w*sw:w*sw+kw] * dZ[
                        i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if padding == 'same':
            da_prev = da_prev[h*sh:h*sh+kh, w*sw:w*sw+kw]
        dA_prev[i, :, :, :] = da_prev

    return dA_prev, dW, db
