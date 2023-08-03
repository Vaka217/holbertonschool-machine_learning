def poly_derivative(poly):
    if not isinstance(poly, list) or len(poly) < 2:
        return None

    derivative = [x * poly[x] for x in range(1, len(poly))]

    return derivative
