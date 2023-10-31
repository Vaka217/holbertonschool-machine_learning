#!/usr/bin/env python3
"""Multinormal Module"""
import numpy as np
mean_cov = __import__("0-mean_cov").mean_cov


class MultiNormal:
    """Represents a Multivariate Normal distribution:

    data is a numpy.ndarray of shape (d, n) containing the data set:
    n is the number of data points
    d is the number of dimensions in each data point
    If data is not a 2D numpy.ndarray, raise a TypeError with the message
    data must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError with the message data must contain
    multiple data points
    Set the public instance variables:
    mean - a numpy.ndarray of shape (d, 1) containing the mean of data
    cov - a numpy.ndarray of shape (d, d) containing the covariance matrix data
    """

    def __init__(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("X must be a 2D numpy.ndarray")

        if data.shape[0] < 2:
            raise ValueError("X must contain multiple data points")

        self.mean, self.cov = mean_cov(data)
