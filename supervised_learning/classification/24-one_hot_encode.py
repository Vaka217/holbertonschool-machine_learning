#!/usr/bin/env python3
"""One-Hot Encode Module"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    return np.array([[1 if y == i else 0 for y in Y] for i in range(classes)])
