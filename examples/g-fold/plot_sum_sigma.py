#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


def main():
    feasible_data = np.genfromtxt(
        "feasible_sum_sigma_vs_n.csv",
        dtype=float,
        delimiter=",",
        names=True,
        deletechars="",
    )
    feasible_Ns = feasible_data["N"]
    feasible_sum_σs = feasible_data["sum(σ)"]

    infeasible_data = np.genfromtxt(
        "infeasible_sum_sigma_vs_n.csv",
        dtype=float,
        delimiter=",",
        names=True,
        deletechars="",
    )
    infeasible_Ns = infeasible_data["N"]
    infeasible_sum_σs = infeasible_data["sum(σ)"]

    plt.figure()
    ax = plt.gca()
    ax.set_title("sum(σ) vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("sum(σ)")
    ax.scatter(
        feasible_Ns, feasible_sum_σs, color="green", marker=".", label="Feasible"
    )
    ax.scatter(
        infeasible_Ns, infeasible_sum_σs, c="red", marker=".", label="Infeasible"
    )
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
