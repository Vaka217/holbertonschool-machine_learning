#!/usr/bin/env python3
"""Sum total"""


def summation_i_squared(n):
    """Calculates the sum from 1 to n of i^2"""
    if not isinstance(n, int) or n < 1:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
