#!/usr/bin/env python3
"""PCA Module"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that the PCA transformation should
    maintain
    Returns: the weights matrix, W, that maintains var fraction of X‘s original
    variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of
    the transformed X
    """
    U, S, Vh = np.linalg.svd(X)

    V = Vh.T

    cumsum = np.cumsum(S)
    cumsum /= cumsum[-1]

    r = np.where(cumsum >= var)[0][0]

    return V[:, :r + 1]
