#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None

    mat = [[sum(mat1[z][j] * mat2[j][i] for j in range(len(mat1[0])))
            for i in range(len(mat2[0]))] for z in range(len(mat1))]

    return mat
