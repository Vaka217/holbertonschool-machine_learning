#!/usr/bin/env python3
"""One-Hot Decode Module"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    return np.array([np.argmax(column) for column in one_hot])
