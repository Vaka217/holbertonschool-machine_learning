#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""
import numpy as np
from math import ceil, floor


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a same convolution on grayscale images

    Args:
        images (numpy.ndarray): ndarray with shape (m, h, w) containing
        multiple grayscale images.
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images

        kernel (numpy.ndarray): ndarray with shape (kh, kw) containing the
        kernel for the convolution.
            - kh is the height of the kernel
            - kw is the width of the kernel

        padding (tuple/str): either a tuple of (ph, pw), ‘same’, or ‘valid’
            - ph is the padding for the height of the image
            - pw is the padding for the width of the image

        stride (tuple): tuple of (sh, sw)

    Returns:
        numpy.ndarray: contains the convolved images
    """

    kh, kw = kernel.shape

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'valid':
        ph = min((kh - 1) // 2, kh // 2)
        pw = min((kw - 1) // 2, kw // 2)

    if padding != 'same':
        images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)])

    m, h, w = images.shape

    sh, sw = stride
    outh = min((h - kh + 1) / sh, (h - kh) / sh)
    outw = min((w - kw + 1) / sw, (h - kw) / sw)

    output = np.zeros((m, outh, outw))

    for i in range(outh):
        for j in range(outw):
            image_section = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_section * kernel, axis=(1, 2))

    return output
