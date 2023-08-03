def summation_i_squared(n):
    if not isinstance(n, int) or n < 1:
        return None

    if n == 1:
        return 1

    return n** 2 + summation_i_squared((n - 1))
