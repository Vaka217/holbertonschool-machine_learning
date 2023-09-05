#!/usr/bin/env python3
"""Forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation using Dropout

    Arguments:
    X -- input dataset
    weights -- weights and biases of the NN
    L -- number of layers in the NN
    keep_prob -- probability that a node will be kept

    Returns:
    cache -- a dictionary containing "Z", "A", "W" and "B"
    """

    cache = {"A0": X}
    t = []
    for i in range(1, L):
        Z = np.dot(weights["W{}".format(i)], cache["A{}".format(
            i - 1)]) + weights["b{}".format(i)]
        cache["A{}".format(i)] = np.tanh(Z)
        cache["D{}".format(i)] = (np.random.rand(
            cache["A{}".format(i)].shape[0],
            cache["A{}".format(i)].shape[1]) < keep_prob).astype(int)
        cache["A{}".format(i)] *= cache["D{}".format(i)]
        cache["A{}".format(i)] /= keep_prob
    Z = np.dot(weights["W{}".format(L)],
               cache["A{}".format(L - 1)]) + weights["b{}".format(L)]
    t = np.exp(Z)
    cache["A{}".format(L)] = t / np.sum(t, axis=0, keepdims=True)
    return cache
