#!/usr/bin/env python3
"""Variable with Adam"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable using Adam optimization algorithm"""

    v = (beta1 * v + (1 - beta1) * grad) / (1 - (beta1 ** t))
    s = (beta2 * s + (1 - beta2) * (grad ** 2)) / (1 - (beta2 ** t))
    var -= alpha * v / (s ** 0.5 + epsilon)
    return var, v, s
