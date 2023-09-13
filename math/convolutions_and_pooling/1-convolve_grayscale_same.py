#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_same(images, kernel):
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

    Returns:
        numpy.ndarray: contains the convolved images
    """

    kh, kw = kernel.shape

    ph = ceil(kh / 2 - 1)
    pw = ceil(kw / 2 - 1)

    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)])
    m, h, w = images.shape

    outh = h - kh + 1
    outw = w - kw + 1

    output = np.zeros((m, outh, outw))

    for i in range(outh):
        for j in range(outw):
            image_section = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_section * kernel, axis=(1, 2))

    return output
