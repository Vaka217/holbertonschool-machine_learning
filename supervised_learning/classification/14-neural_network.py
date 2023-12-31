#!/usr/bin/env python3
"""Neural Network Module 6"""
import numpy as np


class NeuralNetwork():
    """Neural Network Class"""
    def __init__(self, nx, nodes):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Get W1"""
        return self.__W1

    @property
    def b1(self):
        """Get b1"""
        return self.__b1

    @property
    def A1(self):
        """Get A1"""
        return self.__A1

    @property
    def W2(self):
        """Get W2"""
        return self.__W2

    @property
    def b2(self):
        """Get b2"""
        return self.__b2

    @property
    def A2(self):
        """Get A2"""
        return self.__A2

    def forward_prop(self, X):
        """Forward Propagation"""
        Z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        return -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluates the NN Predictions"""
        _, A2 = self.forward_prop(X)
        P = np.where(A2 > 0.5, 1, 0)
        C = self.cost(Y, A2)
        return P, C

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * A1 * (1 - A1)
        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

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
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        return self.evaluate(X, Y)
