#!/usr/bin/env python3

matrix_shape = __import__('2-size_me_please').matrix_shape

def add_matrices2D(mat1, mat2):
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    arr = [[mat1[i][j] + mat2[i][j] for i in range(len(mat1))] for j in range(len(mat1[0]))]
    return arr
