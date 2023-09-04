#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""

    # Creates numpy matrix for confusion of shape(10, 10)
    classes = len(labels[1])
    confusion = np.zeros((classes, classes))

    # Iterates each row, i and j being the correct number and the predicted
    # number respectively. If i == j that means the prediction is correct
    for m in range(len(labels)):
        i = int(np.where(labels[m] == 1)[0])
        j = int(np.where(logits[m] == 1)[0])
        confusion[i][j] += 1

    return (confusion)
