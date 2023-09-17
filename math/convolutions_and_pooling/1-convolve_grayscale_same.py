#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""
import numpy as np


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

    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    m, h, w = images.shape
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)])

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_section = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_section * kernel, axis=(1, 2))

    return output
