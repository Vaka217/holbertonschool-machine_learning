#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer of a
neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a
    neural network.

    Args:
        A_prev (numpy.ndarray): ndarray with shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer.
            - m is the number of examples.
            - h_prev is the height of the previous layer.
            - w_prev is the width of the previous layer.
            - c_prev is the number of channels in the previous layer.

        W (numpy.ndarray): ndarray with shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution.
            - kh is the filter height.
            - kw is the filter width.
            - c_prev is the number of channels in the previous layer.
            - c_new is the number of channels in the output.

        padding (str): either 'same' or 'valid', indicating the type of
        padding used.

        stride (tuple): tuple of (sh, sw) containing the strides for the
        convolution.
            - sh is the stride for the height.
            - sw is the stride for the width.

    Returns:
        numpy.ndarray: the output of the convolutional layer.
    """

    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    ph, pw = 0, 0

    if padding == 'same':
        ph = (((h_prev - 1) * sh + kh - h_prev) // 2) + 1
        pw = (((w_prev - 1) * sw + kw - w_prev) // 2) + 1
        A_prev = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)])

    nh = (h_prev + (2 * ph) - kh) // sh + 1
    nw = (w_prev + (2 * pw) - kw) // sw + 1

    output = np.zeros((m, nh, nw, c_new))

    for channel in range(c_new):
        kernel = W[:, :, :, channel]
        for i in range(nh):
            for j in range(nw):
                image_section = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
                output[:, i, j, channel] = np.sum(image_section * kernel,
                                                  axis=(1, 2, 3))
    output += b

    return activation(output)
