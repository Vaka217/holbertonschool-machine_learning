#!/usr/bin/env python3
"""Normalization of constants"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix"""
    n = X.shape[0]
    m = sum(X) / len(X)
    s = np.sqrt(sum((X - m) ** 2) / len(X))
    return m, s
