#!/usr/bin/env python3
"""Performs K-means Module"""
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed
    If no change in the cluster centroids occurs between iterations, your
    function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    (based on0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
    its centroid

    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to
    """

    C = initialize(X, k)
    n, d = X.shape

    for i in range(iterations):
        C_prev = np.copy(C)

        D = X - C[:, np.newaxis, np.newaxis]
        print(D)

        # juan = np.array(D_all).argmin(axis=1)

        # pedro = [0, 0, 0, 0, 0]
        # centroids = np.zeros((k, d))
        # for j in juan:
        #     centroids[j] += X[j]
        #     pedro[j] += 1

        # pedro = np.array(pedro)
        # print(pedro)
        # centroids = centroids.T / pedro

        # if np.array_equal(centroids.T, centrods):
        #     return centroids.T, 1

        # centrods = centroids.T

    return 0, 1
