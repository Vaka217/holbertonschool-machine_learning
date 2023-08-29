#!/usr/bin/env python3
"""Normalization of matrix"""
import numpy as np


def normalize(X, m, s):
    """Normalizes (standarizes) a matrix"""
    return (X - m) / s
