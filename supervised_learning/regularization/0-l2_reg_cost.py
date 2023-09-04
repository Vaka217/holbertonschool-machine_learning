#!/usr/bin/env python3
"""Calculates the cost of NN with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of NN with L2 regularization

    Arguments:
        cost -- cost of NN
        lambtha -- regularization parameter
        weights -- weights of NN
        L -- number of layers
        m -- number of neurons in each layer

    Returns:
        l2_cost -- cost of NN with L2 regularization
    """
    # Calculate the sum of the weights power by 2 of each layer
    W = [np.sum(np.square(weights['W' + str(i)])) for i in range(1, L + 1)]

    # Calculate the cost of NN with L2 regularization
    l2_cost = cost + lambtha / (2 * m) * sum(W)

    return l2_cost
