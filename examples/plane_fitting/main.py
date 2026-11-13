#!/usr/bin/env python

"""
Finds the floor plane in a depth camera's point cloud that also contains points
on top of a box sitting on the floor.

The plane is nᵀp = d, where n is the unit normal and d is the signed distance
from the camera to the plane. n is a sphere decision variable, which the solver
keeps unit length. A robust cost ignores the box's points, and a constraint
keeps the normal near vertical so the fit can't pick a wall.
"""

import math

import numpy as np
from sleipnir.autodiff import Variable, log
from sleipnir.optimization import Problem, sphere


def main() -> None:
    problem = Problem()

    # Points in the camera frame (x forward, y left, z up) in meters. The first
    # 16 are on the floor with 5 mm of noise. The last 5 are on top of a 0.25 m
    # tall box.
    #
    # The floor's true normal is (0.0498, −0.0796, 0.9956) and d = −0.4 m. A
    # least squares fit of every point gets (0.0648, −0.1279, 0.9897), 2.9° off,
    # and d = −0.322 m, because the box's points pull the plane up toward them.
    points = np.array(
        [
            [0.417, -0.953, -0.497],
            [0.516, -0.381, -0.459],
            [0.496, 0.232, -0.412],
            [0.423, 0.878, -0.354],
            [0.986, -0.882, -0.527],
            [1.091, -0.343, -0.485],
            [1.039, 0.259, -0.433],
            [1.095, 0.860, -0.390],
            [1.579, -0.884, -0.542],
            [1.555, -0.394, -0.506],
            [1.475, 0.218, -0.460],
            [1.586, 0.841, -0.413],
            [1.960, -0.852, -0.563],
            [1.944, -0.234, -0.516],
            [2.036, 0.364, -0.480],
            [2.052, 0.975, -0.419],
            [1.100, 0.250, -0.182],
            [1.300, 0.250, -0.195],
            [1.100, 0.450, -0.164],
            [1.300, 0.450, -0.181],
            [1.200, 0.350, -0.187],
        ]
    )

    # Plane normal and offset. The initial guess is a level floor at the points'
    # mean height.
    n = problem.decision_variable(sphere, 3)
    n.set_value(np.array([0.0, 0.0, 1.0]))

    d = problem.decision_variable()
    d.set_value(points[:, 2].mean())

    # Cauchy loss of each point's distance from the plane. Points much farther
    # than c from the plane barely affect the cost, so the box's points don't
    # pull the fit toward them.
    c = 0.02
    J = Variable(0.0)
    for p in points:
        r = n.dot(p) - d
        J += c**2 * log(1 + r**2 / c**2)
    problem.minimize(J)

    # The normal is within 15° of vertical
    problem.subject_to(n[2] >= math.cos(math.radians(15)))

    problem.solve(diagnostics=True)

    nx, ny, nz = n.value()
    print(f"n = ({nx}, {ny}, {nz})")
    print(f"d = {d.value()} m")


if __name__ == "__main__":
    main()
