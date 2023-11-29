#!/usr/bin/env python3
"""Gaussian Process Module"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process:

    Class constructor: def __init__(self, X_init, Y_init, l=1, sigma_f=1):

    X_init is a numpy.ndarray of shape (t, 1) representing the inputs already
    sampled with the black-box function
    Y_init is a numpy.ndarray of shape (t, 1) representing the outputs of the
    black-box function for each input in X_init
    t is the number of initial samples
    l is the length parameter for the kernel
    sigma_f is the standard deviation given to the output of the black-box
    function
    Sets the public instance attributes X, Y, l, and sigma_f corresponding to
    the respective constructor inputs
    Sets the public instance attribute K, representing the current covariance
    kernel matrix for the Gaussian process

    Public instance method def kernel(self, X1, X2): that calculates the
    covariance kernel matrix between two matrices

    Public instance method def predict(self, X_s): that predicts the mean
    and standard deviation of points in a Gaussian process

    Public instance method def update(self, X_new, Y_new): that updates a
    Gaussian Process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices:

        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)

        Returns: the covariance kernel matrix as a numpy.ndarray of
        shape (m, n)"""

        numerator = -(np.abs(X1 - X2.T) ** 2)
        return (self.sigma_f ** 2) * (np.exp(numerator / (2 * self.l ** 2)))

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in a Gaussian
        process:
        X_s is a numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated
        s is the number of sample points

        Returns: mu, sigma
        mu is a numpy.ndarray of shape (s,) containing the mean for each point
        in X_s, respectively
        sigma is a numpy.ndarray of shape (s,) containing the variance for each
        point in X_s, respectively
        """

        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mean_s = K_s.T.dot(K_inv).dot(self.Y)
        sigma_s = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mean_s.reshape(mean_s.shape[0]), sigma_s

    def update(self, X_new, Y_new):
        """Updates a Gaussian Process:
        X_new is a numpy.ndarray of shape (1,) that represents the new sample
        point
        Y_new is a numpy.ndarray of shape (1,) that represents the new sample
        function value

        Updates the public instance attributes X, Y, and K
        """

        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
