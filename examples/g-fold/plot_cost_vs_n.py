#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


def main():
    feasible_data = np.genfromtxt(
        "feasible_cost_vs_n.csv",
        dtype=float,
        delimiter=",",
        names=True,
        deletechars="",
    )
    feasible_Ns = feasible_data["N"]
    feasible_costs = feasible_data["Cost"]

    infeasible_data = np.genfromtxt(
        "infeasible_cost_vs_n.csv",
        dtype=float,
        delimiter=",",
        names=True,
        deletechars="",
    )
    infeasible_Ns = infeasible_data["N"]
    infeasible_costs = infeasible_data["Cost"]

    plt.figure()
    ax = plt.gca()
    ax.set_title("Cost vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("Cost")
    ax.scatter(feasible_Ns, feasible_costs, color="green", marker=".", label="Feasible")
    ax.scatter(infeasible_Ns, infeasible_costs, c="red", marker=".", label="Infeasible")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
