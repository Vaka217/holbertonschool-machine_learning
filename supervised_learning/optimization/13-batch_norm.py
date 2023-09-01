#!/usr/bin/env python3
"""Batch Norm"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
    normalization"""
    m = np.mean(Z, axis=0)
    v = np.var(Z, axis=0)
    Z_normal = (Z - m) / np.sqrt(v + epsilon)
    return gamma * Z_normal + beta
