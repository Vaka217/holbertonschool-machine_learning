#!/usr/bin/env python3
"""Baum-Welch Algorithm Module"""
import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model:

    Observations is a numpy.ndarray of shape (T,) that contains the index of
    the observation
    T is the number of observations
    Transition is a numpy.ndarray of shape (M, M) that contains the initialized
    transition probabilities
    M is the number of hidden states
    Emission is a numpy.ndarray of shape (M, N) that contains the initialized
    emission probabilities
    N is the number of output states
    Initial is a numpy.ndarray of shape (M, 1) that contains the initialized
    starting probabilities
    iterations is the number of times expectation-maximization should be
    performed

    Returns: the converged Transition, Emission, or None, None on failure
    """
    M, N = Transition.shape
    T = len(Observations)

    for _ in range(iterations):
        P, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)

        gamma = np.zeros((M, T))
        xi = np.zeros((M, M, T - 1))

        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = alpha[i, t] * Transition[i, j] * \
                        Emission[j, Observations[t + 1]]
            xi[:, :, t] /= np.sum(xi[:, :, t])
            gamma[:, t] = np.sum(xi[:, :, t], axis=1)

        gamma[:, T - 1] = alpha[:, T - 1] / P

        for i in range(M):
            for k in range(N):
                Emission[i, k] = np.sum(gamma[i, Observations == k]) / \
                    np.sum(gamma[i, :])

        for i in range(M):
            for j in range(M):
                Transition[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :-1])

    return Transition, Emission
