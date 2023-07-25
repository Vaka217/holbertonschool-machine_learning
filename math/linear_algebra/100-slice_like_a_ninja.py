#!/usr/bin/env python3
"""Write a function that slices a matrix along specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    slice_matrix = matrix.copy()
    for axis in axes:
        slices = tuple([slice(None)] * axis) + (slice(*axes[axis]),)
        slice_matrix = slice_matrix[slices]
    return slice_matrix
