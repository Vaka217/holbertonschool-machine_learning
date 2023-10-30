#!/usr/bin/env python3
"""Definiteness Module"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix:

    matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
    calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the message matrix
    must be a numpy.ndarray
    matrix is not a valid matrix, return None
    Return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite if the matrix
    is positive definite, positive semi-definite, negative semi-definite,
    negative definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] < 1 or not matrix.shape[0] == matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if all(ev > 0 for ev in eigenvalues):
        return ("Positive definite")
    elif all(ev >= 0 for ev in eigenvalues):
        return ("Positive semi-definite")
    elif all(ev <= 0 for ev in eigenvalues):
        return ("Negative semi-definite")
    elif all(ev < 0 for ev in eigenvalues):
        return ("Negative definite")
    else:
        return ("Indefinite")
