#!/usr/bin/env python3
"""Updates weights and biases of NN using gradient descent with
L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases of NN using gradient descent with
    L2 regularization

    Arguments:
        Y -- labels for the data
        weights -- weights and biases of the NN
        cache -- outputs of each layer of the NN
        alpha -- learning rate
        lambtha -- L2 regularization parameter
        L -- number of layers
    """

    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        dW = (1 / m * np.dot(dZ, cache["A{}".format(i - 1)].T)
              ) + lambtha / m * weights["W{}".format(i)]
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.dot(weights["W{}".format(i)].T, dZ) * (1 - cache[
            "A{}".format(i - 1)] ** 2)
        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db
