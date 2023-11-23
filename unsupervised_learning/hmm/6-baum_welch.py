#!/usr/bin/env python3
"""Baum-Welch Algorithm Module"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden Markov model:

    Observation is a numpy.ndarray of shape (T,) that contains the index of the
    observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
    Transition[i, j] is the probability of transitioning from the hidden state
    i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state

    Returns: P, F, or None, None on failure
    P is the likelihood of the observations given the model
    F is a numpy.ndarray of shape (N, T) containing the forward path
    probabilities
    F[i, j] is the probability of being in hidden state i at time j given the
    previous observations
    """

    alpha_t = np.zeros((len(Initial), len(Observation)))
    alpha_t[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, len(Observation)):
        for j in range(len(Initial)):
            alpha_t_j = np.sum(
                alpha_t[:, t - 1] * Transition[:, j]) * \
                Emission[j, Observation[t]]
            alpha_t[j, t] = alpha_t_j

    P = np.sum(alpha_t[:, -1])

    return P, alpha_t


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden Markov model:

    Observation is a numpy.ndarray of shape (T,) that contains the index of the
    observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
    Transition[i, j] is the probability of transitioning from the hidden state
    i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state

    Returns: P, B, or None, None on failure
    P is the likelihood of the observations given the model
    F is a numpy.ndarray of shape (N, T) containing the backward path
    probabilities
    B[i, j] is the probability of generating the future observations from
    hidden state i at time j
    """

    beta_t = np.zeros((len(Initial), len(Observation)))
    beta_t[:, -1] = 1

    for t in range(len(Observation) - 2, -1, -1):
        for j in range(len(Initial)):
            beta_t[j, t] = np.sum(
                beta_t[:, t + 1] * Transition[j, :] *
                Emission[:, Observation[t + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta_t[:, 0])
    return P, beta_t


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
                        Emission[j, Observations[t + 1]] * beta[j, t + 1]
            xi[:, :, t] /= np.sum(xi[:, :, t])
            gamma[:, t] = np.sum(xi[:, :, t], axis=1)

        gamma[:, T - 1] = alpha[:, T - 1] * beta[:, T - 1] / P

        for i in range(M):
            for k in range(N):
                Emission[i, k] = np.sum(gamma[i, Observations == k]) / \
                    np.sum(gamma[i, :])

        for i in range(M):
            for j in range(M):
                Transition[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :])

    return Transition, Emission
