#!/usr/bin/env python3
"""Deep Neural Network Module 0"""
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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            self.weights["W{}".format(l)] = np.random.randn(
                layers[l-1], nx) * np.sqrt(2 / nx)
            self.weights["b{}".format(l)] = np.zeros((layers[l-1], 1))
            nx = layers[l-1]
