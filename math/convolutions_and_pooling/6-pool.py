#!/usr/bin/env python3
"""Performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images

    Args:
        images (numpy.ndarray): ndarray with shape (m, h, w, c) containing
        multiple images.
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image

        kernel (tuple): is a tuple of (kh, kw) containing the kernel
        shape for the pooling
            - kh is the height of the kernel
            - kw is the width of the kernel

        stride (tuple): tuple of (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image

        mode (str): type of pooling
            - max indicates max pooling
            - avg indicates average pooling
    Returns:
        numpy.ndarray: contains the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    outh = ((h - kh) // sh) + 1
    outw = ((w - kw) // sw) + 1

    output = np.zeros((m, outh, outw, c))

    for i in range(outh):
        for j in range(outw):
            image_section = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            if mode == 'max':
                output[:, i, j, :] = np.max(image_section, axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.average(image_section, axis=(1, 2))

    return output
