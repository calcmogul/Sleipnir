#!/usr/bin/env python

"""
Determines a robot pose from the corner pixel locations of several AprilTags.

The robot pose is constrained to be on the floor (z = 0).
"""

import numpy as np
from sleipnir.autodiff import Translation2, VariableMatrix
from sleipnir.optimization import Problem, se2


def main() -> None:
    problem = Problem()

    # camera calibration
    fx = 600
    fy = 600
    cx = 300
    cy = 150

    # robot pose. The robot is on the floor, so its pose is 2D.
    field2robot = problem.decision_variable(se2)

    # robot is ENU, cameras are SDE
    robot2camera = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    # list of points in field space to reproject. Each one is a 3x1 vector of
    # (x,y,z)
    field2points = [
        np.array([2, 0 - 0.08255, 0.4]),
        np.array([2, 0 + 0.08255, 0.4]),
    ]

    # List of points we saw the target at. These are exactly what we expect for
    # a camera located at 0,0,0 (hand-calculated)
    point_observations = [(325, 30), (275, 30)]

    # initial guess at robot pose. We expect the robot to converge to 0,0,0
    field2robot.translation().set_value(np.array([-0.1, 0.0]))
    field2robot.rotation().set_value(0.2)

    robot2field = field2robot.inverse()

    # Cost
    J = 0
    for field2point, observation in zip(field2points, point_observations):
        # point's coordinates in robot frame. The robot is on the floor, so
        # only the point's x and y change.
        robot2point_xy = Translation2(field2point[:2]).transform_by(robot2field)
        robot2point = VariableMatrix(
            [[robot2point_xy.x()], [robot2point_xy.y()], [field2point[2]]]
        )

        # point's coordinates in camera frame
        camera2point = robot2camera.T @ robot2point
        x = camera2point[0]
        y = camera2point[1]
        z = camera2point[2]

        print(f"camera2point = {x.value()}, {y.value()}, {z.value()}")

        # coordinates observed at
        u_observed, v_observed = observation

        X = x / z
        Y = y / z

        u = fx * X + cx
        v = fy * Y + cy

        print(f"Expected u {u.value()}, saw {u_observed}")
        print(f"Expected v {v.value()}, saw {v_observed}")

        u_err = u - u_observed
        v_err = v - v_observed

        # Cost function is square of reprojection error
        J += u_err**2 + v_err**2

    problem.minimize(J)

    problem.solve(diagnostics=True)

    x, y = field2robot.translation().value()
    print(f"x = {x} m")
    print(f"y = {y} m")
    print(f"θ = {field2robot.rotation().value()} rad")


if __name__ == "__main__":
    main()
