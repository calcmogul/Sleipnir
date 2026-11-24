#!/usr/bin/env python

"""
Finds the cheapest generator dispatch for the WSCC 9-bus system (MATPOWER's
case9) subject to the AC power flow equations.

Bus voltages V are complex decision variables in rectangular form. The power
injected at each bus is S = V ∘ conj(Y V), where Y is the bus admittance matrix,
and it must equal generation minus load. Voltage magnitudes, generator outputs,
and line apparent power flows are limited. The cost is each generator's
quadratic fuel cost.

Quantities are per unit on a 100 MVA base except where noted.

The data is from MATPOWER's case9:
https://github.com/MATPOWER/matpower/blob/master/data/case9.m

The problem is the standard AC OPF in section 6.1 of the MATPOWER User's Manual
(version 8.1), with Cartesian voltages as in section 6.1.1. Branches use the π
model in section 3.2 with no transformers, and the network equations are in
section 3.6:
https://matpower.org/docs/MATPOWER-manual.pdf

MATPOWER's runopf('case9') solves the same problem.
"""

import math

import numpy as np
from sleipnir.autodiff import Real
from sleipnir.optimization import Problem, bounds


def main() -> None:
    base_MVA = 100.0

    # Load in MW and MVAr at each bus
    P_load = np.array([0, 0, 0, 0, 90, 0, 100, 0, 125])
    Q_load = np.array([0, 0, 0, 0, 30, 0, 35, 0, 50])

    # Generators: bus, P limits in MW, Q limits in MVAr, and cost c₂P² + c₁P + c₀
    # in $/h with P in MW
    generators = [
        # bus, P_min, P_max, Q_min, Q_max, c2, c1, c0
        (0, 10, 250, -300, 300, 0.11, 5, 150),
        (1, 10, 300, -300, 300, 0.085, 1.2, 600),
        (2, 10, 270, -300, 300, 0.1225, 1, 335),
    ]

    # Branches: from bus, to bus, series resistance and reactance, total line
    # charging susceptance, and apparent power limit in MVA
    branches = [
        (0, 3, 0, 0.0576, 0, 250),
        (3, 4, 0.017, 0.092, 0.158, 250),
        (4, 5, 0.039, 0.17, 0.358, 150),
        (2, 5, 0, 0.0586, 0, 300),
        (5, 6, 0.0119, 0.1008, 0.209, 150),
        (6, 7, 0.0085, 0.072, 0.149, 250),
        (7, 1, 0, 0.0625, 0, 250),
        (7, 8, 0.032, 0.161, 0.306, 250),
        (8, 3, 0.01, 0.085, 0.176, 250),
    ]

    n_bus = len(P_load)
    n_gen = len(generators)
    n_branch = len(branches)

    # Each branch is a π model: a series admittance y between its buses and half
    # the line charging to ground at each end. The currents into branch k at its
    # ends are
    #
    #   [I_from,k]   [y + jb/2     −y   ] [V_f]
    #   [I_to,k  ] = [  −y      y + jb/2] [V_t]
    #
    # C_from and C_to select each branch's from and to bus voltages. The current
    # into the branch at each end is I_from = Y_from V and I_to = Y_to V, and the
    # bus admittance matrix Y sums the currents each branch draws from each bus,
    # so I = Y V.
    #
    # Products with constant matrices skip their zero entries, so dense NumPy
    # arrays only build expressions for the nonzeros.
    C_from = np.zeros((n_branch, n_bus))
    C_to = np.zeros((n_branch, n_bus))
    y_series = np.zeros(n_branch, dtype=np.complex128)
    y_shunt = np.zeros(n_branch, dtype=np.complex128)
    for k, (f, t, r, x, b, _) in enumerate(branches):
        C_from[k, f] = 1
        C_to[k, t] = 1
        y_series[k] = 1 / complex(r, x)
        y_shunt[k] = 1j * b / 2

    Y_from = np.diag(y_series + y_shunt) @ C_from - np.diag(y_series) @ C_to
    Y_to = -np.diag(y_series) @ C_from + np.diag(y_series + y_shunt) @ C_to
    Y = C_from.T @ Y_from + C_to.T @ Y_to

    # Maps generator outputs to the buses they inject into
    C_gen = np.zeros((n_bus, n_gen))
    for g, gen in enumerate(generators):
        C_gen[gen[0], g] = 1

    S_load = ((P_load + 1j * Q_load) / base_MVA).reshape(-1, 1)

    problem = Problem()

    # Bus voltages start at 1∠0 p.u., and generator outputs start at zero
    V = problem.complex_decision_variable(n_bus)
    V.set_value(np.ones((n_bus, 1)))
    S_gen = problem.complex_decision_variable(n_gen)

    # Generator fuel cost
    cost = Real(0.0)
    for g, (_, _, _, _, _, c2, c1, c0) in enumerate(generators):
        P = S_gen[g].real() * base_MVA
        cost += c2 * P * P + c1 * P + c0
    problem.minimize(cost)

    # Bus 1's voltage angle is the zero reference
    problem.subject_to(V[0].real() >= 0)
    problem.subject_to(V[0].imag() == 0)

    # Power balance at each bus
    problem.subject_to(V.cwise_product((Y @ V).conj()) == C_gen @ S_gen - S_load)

    # Voltage magnitudes are within 0.9 to 1.1 p.u.
    V_abs2 = V.abs2()
    problem.subject_to(V_abs2 >= 0.9**2)
    problem.subject_to(V_abs2 <= 1.1**2)

    # Generator output limits
    for g, (_, P_min, P_max, Q_min, Q_max, _, _, _) in enumerate(generators):
        problem.subject_to(bounds(P_min / base_MVA, S_gen[g].real(), P_max / base_MVA))
        problem.subject_to(bounds(Q_min / base_MVA, S_gen[g].imag(), Q_max / base_MVA))

    # Apparent power flowing into each end of each branch is within its rating
    S_from = (C_from @ V).cwise_product((Y_from @ V).conj())
    S_to = (C_to @ V).cwise_product((Y_to @ V).conj())
    for k, branch in enumerate(branches):
        rating = branch[5] / base_MVA
        problem.subject_to(S_from[k].abs2() <= rating**2)
        problem.subject_to(S_to[k].abs2() <= rating**2)

    problem.solve(diagnostics=True)

    print(f"cost = {cost.value():.2f} $/h")
    for g in range(n_gen):
        s = S_gen[g].value() * base_MVA
        print(f"generator {g + 1}: P = {s.real:.2f} MW, Q = {s.imag:.2f} MVAr")
    for i in range(n_bus):
        v = V[i]
        print(
            f"bus {i + 1}: |V| = {v.abs().value():.4f} p.u., "
            f"∠V = {math.degrees(v.arg().value()):.3f}°"
        )


if __name__ == "__main__":
    main()
