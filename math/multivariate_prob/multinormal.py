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
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = mean_cov(data.T)

    def pdf(self, x):
        """Calculates the PDF at a data point:
        x is a numpy.ndarray of shape (d, 1) containing the data point whose
        PDF should be calculated
        d is the number of dimensions of the Multinomial instance
        If x is not a numpy.ndarray, raise a TypeError with the message x must
        be a numpy.ndarray
        If x is not of shape (d, 1), raise a ValueError with the message x must
        have the shape ({d}, 1)
        Returns the value of the PDF
        """
        d = len(self.mean)

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        pdf = (1 / (((2 * np.math.pi)**(d/2)) *
                    (np.linalg.det(self.cov)**(1/2)))) * \
            np.exp((-1/2) *
                   ((x-self.mean).T.dot(np.linalg.inv(self.cov
                                                      ))).dot((x-self.mean)))
        return float(pdf)
