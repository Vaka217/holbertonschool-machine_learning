#!/usr/bin/env python3
"""Determinant Module"""


def determinant(matrix):
    """Calculates the determinant of a matrix

    Args:
        matrix: list of lists whose determinant should be calculated

    Returns:
        The determinant of matrix
    """
    if matrix == [[]]:
        return 1

    if not matrix or not isinstance(matrix, list) or not matrix[0] or not isinstance(matrix[0], list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    dim_mat = len(matrix) - 1
    det_mats = [matrix]
    coef = [1]
    det = 0
    for i in range(dim_mat, 0, -1):
        if i == 1:
            for z, det_mat in enumerate(det_mats):
                if z % 2 != 0:
                    det -= coef[z] * (det_mat[0][0] * det_mat[1][1] - det_mat[0][1] * det_mat[1][0])
                else:
                    det += coef[z] * (det_mat[0][0] * det_mat[1][1] - det_mat[0][1] * det_mat[1][0])
        else:
            coef = []
            det_mats = []
            for j in range(i + 1):
                coef.append(matrix[0][j])
                det_mats.append([mat[:j] + mat[j + 1:] for mat in matrix[1:]])

    return det
