#!/usr/bin/env python3
"""Calculates the specificity"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix"""

    # Creates np array of shape(classes, )
    specificity = np.zeros((len(confusion)))

    # Calculate the specificity for each class: TN / (TN + FP)
    for i, classs in enumerate(confusion):
        tn = np.sum(np.delete(np.delete(confusion, i, axis=0), i, axis=1))
        specificity[i] = tn / (tn + sum(confusion.T[i]) - classs[i])

    return specificity
