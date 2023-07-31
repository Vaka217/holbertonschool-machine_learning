#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axs = plt.subplots(3, 2)
fig.tight_layout(rect=[0, 0.03, 1, 0.98])
fig.subplots_adjust(wspace=0.3, hspace=0.8)
fig.suptitle("All in One")

plt.subplot(321)
x0 = list(range(0, 11))
plt.xlim([0, max(x0)])
plt.plot(x0, y0, "r-")

plt.subplot(322)
plt.title("Men's Height vs Weight", fontsize="x-small")
plt.xlabel("Height (in)", fontsize="x-small")
plt.ylabel("Weight (lbs)", fontsize="x-small")
plt.scatter(x1, y1, marker="o", c="m")

plt.subplot(323)
plt.title("Exponential Decay of C-14", fontsize="x-small")
plt.xlabel("Time (years)", fontsize="x-small")
plt.ylabel("Fraction Remaining", fontsize="x-small")
plt.xlim([0, 28650])
plt.yscale("log")
plt.plot(x2, y2)

plt.subplot(324)
plt.xlabel("Time (years)", fontsize="x-small")
plt.ylabel("Fraction Remaining", fontsize="x-small")
plt.title("Exponential Decay of Radioactive Elements", fontsize="x-small")
plt.xlim([0, 20000])
plt.ylim([0, 1])
plt.plot(x3, y31, "r--")
plt.plot(x3, y32, "g-")
plt.legend(["C-14", "Ra-226"])

plt.subplot(313)
plt.xlabel("Grades", fontsize="x-small")
plt.ylabel("Number of Students", fontsize="x-small")
plt.title("Project A", fontsize="x-small")
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xticks(np.arange(0, 101, 10))
plt.hist(student_grades, edgecolor="black", bins=range(0, 110, 10))

plt.savefig("5-all_in_one.png")
