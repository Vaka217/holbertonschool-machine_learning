#!/usr/bin/env python3
"""Polynomial integral"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    return [C] + [int(poly[x] / (x + 1)) if poly[x] / (x + 1)
                  == int(poly[x] / (x + 1)) else poly[x] / (x + 1)
                  for x in range(len(poly))]
