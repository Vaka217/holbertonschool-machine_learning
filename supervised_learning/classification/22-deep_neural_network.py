#!/usr/bin/env python3
"""Deep Neural Network Module 6"""
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
        if not all(map(
                lambda layer: isinstance(layer, int) and layer > 0, layers)):
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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]
        dZ = cache["A{}".format(self.L)] - Y
        for i in range(self.L, 0, -1):
            dW = 1 / m * np.dot(dZ, cache["A{}".format(i - 1)].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dZ = np.dot(self.weights["W{}".format(i)].T, dZ) * cache[
                "A{}".format(i - 1)] * (1 - cache["A{}".format(i - 1)])
            self.weights["W{}".format(i)] -= alpha * dW
            self.weights["b{}".format(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            _, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        return self.evaluate(X, Y)
