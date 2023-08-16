#!/usr/bin/env python3
"""Deep Neural Network Module 4"""
import numpy as np


class DeepNeuralNetwork():
    """Neural Network Class"""
    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all([layer == int(layer) and layer > 0 for layer in layers]):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.L + 1):
            self.weights["W{}".format(l)] = np.random.randn(
                layers[l-1], nx) * np.sqrt(2 / nx)
            self.weights["b{}".format(l)] = np.zeros((layers[l-1], 1))
            nx = layers[l-1]

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def weights(self):
        """Weights"""
        return self.__weights

    @property
    def cache(self):
        """Cache"""
        return self.__cache

    def forward_prop(self, X):
        """Forward Propagation"""
        self.cache["A0"] = X
        for i in range(1, self.L + 1):
            Z = np.dot(self.weights["W{}".format(i)], self.cache["A{}".format(
                i - 1)]) + self.weights["b{}".format(i)]
            self.cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))
        return self.cache["A{}".format(self.L)], self.cache

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        return -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluates the NN Predictions"""
        A, _ = self.forward_prop(X)
        P = np.where(A > 0.5, 1, 0)
        C = self.cost(Y, A)
        return P, C
