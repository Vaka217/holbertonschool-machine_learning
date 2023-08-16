#!/usr/bin/env python3
"""One-Hot Decode Module"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a numeric label vector into a one-hot matrix"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    return np.array([np.argmax(column) for column in one_hot])
