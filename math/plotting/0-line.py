#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = list(range(0, 11))

plt.xlim([0, max(x)])

plt.plot(x, y, "r-")

plt.savefig('0-line.png')
