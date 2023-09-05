#!/usr/bin/env python3
"""Updates weights and biases of NN using gradient descent with
Dropout regularization"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights and biases of NN using gradient descent with
    Dropout regularization

    Arguments:
        Y -- labels for the data
        weights -- weights and biases of the NN
        cache -- outputs of each layer of the NN
        alpha -- learning rate
        keep_prob -- probability that a node will be kept
        L -- number of layers
    """

    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y
    for i in range(L, 1, -1):
        dW = 1 / m * np.dot(dZ, cache["A{}".format(i - 1)].T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.dot(weights["W{}".format(i)].T, dZ) * (
            cache["D{}".format(i - 1)] / keep_prob) * (1 - cache[
                "A{}".format(i - 1)] ** 2)
        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db
