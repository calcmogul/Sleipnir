#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

with open(sys.argv[1], encoding="utf-8") as f:
    labels = f.readline().rstrip().split(",")
data = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)

plt.figure()
ax = plt.gca()
iters = data[:, :1]
for i in [1, 2, 3, 4]:
    # for i in range(1, data.shape[1]):
    ax.plot(data[:, :1], data[:, i : i + 1], label=labels[i])
ax.set_xlabel("Iteration")
ax.legend()
plt.show()
