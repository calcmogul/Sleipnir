#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) == 1:
        print("usage: plot_csv.py [file.csv]...")
        sys.exit(1)
    filenames = sys.argv[1:]

    _, axs = plt.subplots(len(filenames), 1)
    for ax, filename in zip(axs, filenames):
        with open(filename, encoding="utf-8") as f:
            labels = f.readline().rstrip().split(",")
        data = np.genfromtxt(filename, delimiter=",", skip_header=1)

        ax.scatter(data[:, :1], data[:, 1:2], marker="x")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        # ax.set_aspect(1.0)
        # ax.set_box_aspect(1.0)
    plt.show()


if __name__ == "__main__":
    main()
