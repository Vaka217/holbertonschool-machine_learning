#!/usr/bin/env python
"""Neural Network Module 4"""
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
