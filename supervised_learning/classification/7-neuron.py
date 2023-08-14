#!/usr/bin/env python
"""Neuron Module 7"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """Neuron Class"""
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W getter method"""
        return self.__W

    @property
    def b(self):
        """b getter method"""
        return self.__b

    @property
    def A(self):
        """A getter method"""
        return self.__A

    def forward_prop(self, X):
        """Forward Propagation"""
        Z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        return -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluate"""
        A = self.forward_prop(X)
        P = np.where(A > 0.5, 1, 0)
        C = self.cost(Y, A)
        return P, C

    def gradient_descent(self, X, Y, A, alpha=0.5):
        """Gradient Descent"""
        m = Y.shape[1]
        dZ = A - Y
        db = 1 / m * np.sum(dZ)
        dW = 1 / m * np.dot(dZ, X.T)
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be a positive float")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be a positive integer and <= iterations")
        costs = []
        for i in range(iterations + 1):
            self.__A = self.forward_prop(X)
            if verbose:
                if i % step == 0 or i == iterations:
                    print("Cost after {} iterations: {}".format(
                        i, self.cost(Y, self.A)))
                    costs.append(self.cost(Y, self.A))
            self.gradient_descent(X, Y, self.A, alpha)
        if graph:
            itera = np.arange(0, iterations + 1, step)
            plt.plot(itera, costs, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.savefig("plotNeuron.png")
        return self.evaluate(X, Y)
