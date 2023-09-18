#!/usr/bin/env python3
"""Perform backpropagation over a pooling layer of a neural network."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform backpropagation over a pooling layer of a neural network.

    Args:
        dA (numpy.ndarray): Partial derivatives with respect to the output of
        the pooling layer.
            - Shape: (m, h_new, w_new, c_new)
              - m: Number of examples.
              - h_new: Height of the output.
              - w_new: Width of the output.
              - c_new: Number of channels.

        A_prev (numpy.ndarray): Output of the previous layer.
            - Shape: (m, h_prev, w_prev, c)
              - m: Number of examples.
              - h_prev: Height of the previous layer.
              - w_prev: Width of the previous layer.
              - c: Number of channels in the previous layer.

        kernel_shape (tuple): Size of the pooling kernel.
            - Tuple format: (kh, kw)
              - kh: Kernel height.
              - kw: Kernel width.

        stride (tuple): Strides for the pooling operation.
            - Tuple format: (sh, sw)
              - sh: Stride for the height.
              - sw: Stride for the width.

        mode (str): Pooling mode.
            - 'max' for maximum pooling.
            - 'avg' for average pooling.

    Returns:
        numpy.ndarray: Partial derivatives with respect to the previous layer
        a(dA_prev).
            - Shape: (m, h_prev, w_prev, c)
              - m: Number of examples.
              - h_prev: Height of the previous layer.
              - w_prev: Width of the previous layer.
              - c: Number of channels in the previous layer.
    """

    m, h_new, w_new, c_new = dA.shape
    m, _, _, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    if mode == 'max':
                        a_prev_slice = A_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c] += mask * dA[
                            i, h, w, c]
                    if mode == 'avg':
                        average_dA = dA[i, h, w, c] / kh / kw
                        dA_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c] += np.ones((
                            kh, kw)) * average_dA

    return dA_prev
