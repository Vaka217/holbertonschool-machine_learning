#!/usr/bin/env python3
"""Binomial Class"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif not 0 <= p <= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.p = sum(data) / len(data) / 50
            self.n = round(sum(data) / len(data) / self.p)
            self.p = sum(data) / len(data) / self.n

    def pmf(self, k):
        """Calculates the PMF"""
        if k is not int:
            k = int(k)
        if k < 0:
            return 0
        nfact = 1
        kfact = 1
        nkfact = 1
        for i in range(1, self.n + 1):
            nfact = nfact * i
        for i in range(1, k + 1):
            kfact = kfact * i
        for i in range(1, self.n - k + 1):
            nkfact = nkfact * i
        nk = nfact / (kfact * (nkfact))
        return nk * self.p ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """Calculates the CDF"""
        if k is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)
        return cdf
