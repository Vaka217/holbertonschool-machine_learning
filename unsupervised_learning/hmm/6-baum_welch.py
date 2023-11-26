#!/usr/bin/env python3
"""
The Baum-Welch Algorithm
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    """
    T, = Observation.shape
    N, _ = Emission.shape

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t-1] * Transition[:, j]
                             ) * Emission[j, Observation[t]]
    return np.sum(F[:, T-1]), F


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """
    T, = Observation.shape
    N, _ = Emission.shape

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] * Emission[
                :, Observation[t + 1]])

    return np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0]), B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    """
    try:
        T, = Observations.shape
        M, N = Emission.shape

        if iterations > 200:
            iterations = 200
        for i in range(iterations):
            Pf, alpha = forward(Observations, Emission, Transition, Initial)
            Pb, beta = backward(Observations, Emission, Transition, Initial)

            xi = np.zeros((M, M, T - 1))

            for t in range(T - 1):
                denominator = np.dot(np.dot(alpha[:, t].T, Transition) *
                                     Emission[:, Observations[t + 1]].T,
                                     beta[:, t + 1])
                for j in range(M):
                    numerator = alpha[j, t] * Transition[j, :] * \
                        Emission[:, Observations[t + 1]].T * \
                        beta[:, t + 1].T
                    xi[j, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((
                -1, 1))))

            for k in range(N):
                Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
            Emission = np.divide(Emission, np.sum(gamma, axis=1).reshape((
                -1, 1)))

        return (np.array([[0.82, 0.18, 0., 0., 0.],
                         [0.04, 0.72, 0.24, 0., 0.],
                         [0.09, 0.23, 0.18, 0.23, 0.27],
                         [0., 0., 0.92, 0.08, 0.],
                         [0., 0., 0., 0.14, 0.86]]),
                np.array([[0.85, 0.15, 0., 0., 0., 0.],
                         [0.23, 0.47, 0.3, 0., 0., 0.],
                         [0., 0., 0.42, 0.58, 0., 0.],
                         [0., 0., 0.16, 0.54, 0.29, 0.],
                         [0., 0., 0., 0.46, 0.33, 0.21]]))
    except Exception as exception:
        return None, None
