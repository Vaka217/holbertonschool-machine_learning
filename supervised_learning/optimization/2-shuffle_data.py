#!/usr/bin/env python3
"""Normalization of matrix"""
import numpy as np


def shuffle_data(X, Y):
    """Normalizes (standarizes) a matrix"""
    shuffle = np.random.permutation(len(X))
    return X[shuffle], Y[shuffle]
