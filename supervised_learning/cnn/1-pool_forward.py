#!/usr/bin/env python3
"""Performs forward propagation over a pooling layer of a neural network."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a
    neural network.

    Args:
        A_prev (numpy.ndarray): ndarray with shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer.
            - m is the number of examples.
            - h_prev is the height of the previous layer.
            - w_prev is the width of the previous layer.
            - c_prev is the number of channels in the previous layer.

        kernel_shape (tuple): tuple of (kh, kw) containing the size of the
        kernel for the pooling.
            - kh is the kernel height.
            - kw is the kernel width.

        stride (tuple): tuple of (sh, sw) containing the strides for the
        pooling.
            - sh is the stride for the height.
            - sw is the stride for the width.

        mode (str): either 'max' or 'avg',  indicating whether to perform
        maximum or average pooling, respectively.

    Returns:
        numpy.ndarray: the output of the pooling layer.
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    outh = ((h_prev - kh) // sh) + 1
    outw = ((w_prev - kw) // sw) + 1

    output = np.zeros((m, outh, outw, c_prev))

    for i in range(outh):
        for j in range(outw):
            image_section = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            if mode == 'max':
                output[:, i, j, :] = np.max(image_section, axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.average(image_section, axis=(1, 2))

    return output
