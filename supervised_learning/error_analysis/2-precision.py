#!/usr/bin/env python3
"""Calculates the precision"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix"""

    # Creates np array of shape(classes, )
    precision = np.zeros((len(confusion)))

    # Calculate the precision for each class: TP / (TP + FP)
    for i, classs in enumerate(confusion):
        precision[i] = classs[i] / sum(confusion.T[i])

    return precision
