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
    cov = np.dot(X.T, X) / (X.shape[0] - 1)
    eigen_values, eigen_vectors = np.linalg.eig(cov)

    i = np.argsort(eigen_values, axis=0)[::-1]
    sorted_eigen_vectors = eigen_vectors[:, i]

    cumsum = np.cumsum(eigen_values[i]) / np.sum(eigen_values[i])
    r = np.sum(np.where(cumsum <= var))
    print(r)
    return sorted_eigen_vectors[:, :r]
