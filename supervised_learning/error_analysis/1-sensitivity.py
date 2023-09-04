#!/usr/bin/env python3
"""Calculates the sensitivity"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""

    # Creates np array of shape(classes, )
    sensitivity = np.zeros((len(confusion)))

    # Calculate the sensitivity for each class: TP / (TP + FN)
    for i, classs in enumerate(confusion):
        sensitivity[i] = classs[i] / sum(classs)

    return sensitivity
