#!/usr/bin/env python3
"""Sum total"""


def summation_i_squared(n):
    """Calculates the sum from 1 to n of i^2"""
    if not isinstance(n, int) or n < 1:
        return None

    if n == 1:
        return 1

    return n ** 2 + summation_i_squared((n - 1))
