#!/usr/bin/env python3
"""Absorbing Chains Module"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
    standard transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain

    Returns: True if it is absorbing, or False on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False

    num_absorbing = [p[i] == 1 for i, p in enumerate(P)]

    if True not in num_absorbing:
        return False

    return True
