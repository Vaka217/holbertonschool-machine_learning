#!/usr/bin/env python3
"""Write a function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    if matrix == []:
        return
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
