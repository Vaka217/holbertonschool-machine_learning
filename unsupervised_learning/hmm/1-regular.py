#!/usr/bin/env python3
"""Regular Chains Module"""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2 or \
            np.count_nonzero(P == 0):
        return None

    eigen_values, eigen_vectors = np.linalg.eig(P)

    eigen_one_idxs = [round(eigen.real) for eigen in eigen_values].index(1.0)

    s = (eigen_vectors[:, eigen_one_idxs]
         / np.sum(eigen_vectors[:, eigen_one_idxs], axis=0))
    prev_s = 0

    while not (np.array_equal(s, prev_s)):
        prev_s = s
        s = prev_s @ P

    return s.reshape(1, len(P))
