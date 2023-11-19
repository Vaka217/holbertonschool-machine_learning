#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster
    You may use at most 1 loop

    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the updated priors for each
    cluster
    m is a numpy.ndarray of shape (k, d) containing the updated centroid means
    for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
    matrices for each cluster
    """

    n, d = X.shape
    k = g.shape[0]

    m, _ = kmeans(X.T, k)

    pi = np.sum(g, axis=1) / n

    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        weighted_diff = (g[i, :, np.newaxis] * diff).T
        S[i] = np.dot(weighted_diff, diff) / np.sum(g[i])

    return pi, m, S
