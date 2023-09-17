#!/usr/bin/env python3
"""Performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images

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
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image

    Returns:
        numpy.ndarray: contains the convolved images
    """

    kh, kw = kernel.shape
    m, h, w = images.shape
    sh, sw = stride
    ph, pw = 0, 0

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1

    if pw and ph:
        images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)])

    outh = ((h + (2 * ph) - kh) // sh) + 1
    outw = ((w + (2 * pw) - kw) // sw) + 1

    output = np.zeros((m, outh, outw))

    for i in range(outh):
        for j in range(outw):
            image_section = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output[:, i, j] = np.sum(image_section * kernel, axis=(1, 2))

    return output
