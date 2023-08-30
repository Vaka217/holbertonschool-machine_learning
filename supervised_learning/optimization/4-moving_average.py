#!/usr/bin/env python3
"""Weighted moving average with bias correction"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""
    v = 0
    move_average = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        move_average.append(v / (1 - beta**(i + 1)))
    return move_average
