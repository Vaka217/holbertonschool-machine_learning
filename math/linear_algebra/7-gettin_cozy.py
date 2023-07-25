#!/usr/bin/env python3
import copy

def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0 and len(mat1[0]) != len(mat2[0]) or axis == 1 and len(mat1) != len(mat2):
        return None
    
    mat = copy.deepcopy(mat1 + mat2) if axis == 0 else [mat1[i] + mat2[i] for i in range(len(mat1))]

    return mat
