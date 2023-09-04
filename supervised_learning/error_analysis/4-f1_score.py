#!/usr/bin/env python3
"""Calculates the F1 score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""

    # Creates np array of shape(classes, )
    f1_score = np.zeros((len(confusion)))

    # Calculate sensitivity and precision
    sensitivity_array = sensitivity(confusion)
    precision_array = precision(confusion)

    # Calculate the F1 score for each class:
    # 2 / (sensitivity ^ -1 + precision ^ -1)
    for i in range(len(confusion)):
        f1_score[i] = 2 / (sensitivity_array[i] ** -
                           1 + precision_array[i] ** -1)

    return f1_score
