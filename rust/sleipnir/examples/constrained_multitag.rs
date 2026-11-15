// Copyright (c) Sleipnir contributors

//! Determines a robot pose from the corner pixel locations of several
//! AprilTags.
//!
//! The robot pose is constrained to be on the floor (z = 0).

// Greek identifiers match the math
#![allow(mixed_script_confusables, confusable_idents)]

use sleipnir::{Options, Problem, Variable, VariableMatrix};

fn main() {
    let problem: Problem = Problem::new();

    // Camera calibration
    let (fx, fy) = (600.0, 600.0);
    let (cx, cy) = (300.0, 150.0);

    // Robot pose
    let robot_x = problem.decision_variable();
    let robot_y = problem.decision_variable();
    let robot_z = 0.0;
    let robot_θ = problem.decision_variable();

    // Cache autodiff variables
    let sinθ = robot_θ.sin();
    let cosθ = robot_θ.cos();

    let zero = Variable::from(0.0);
    let one = Variable::from(1.0);
    let field2robot = VariableMatrix::from([
        [cosθ, -sinθ, zero, robot_x],
        [sinθ, cosθ, zero, robot_y],
        [zero, zero, one, robot_z.into()],
        [zero, zero, zero, one],
    ]);

    // Robot is ENU, cameras are SDE
    let robot2camera = [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let field2camera = &field2robot * robot2camera;

    // List of points in field space to reproject. Each one is a 4x1 vector of
    // (x, y, z, 1).
    let field2points = [
        [[2.0], [0.0 - 0.08255], [0.4], [1.0]],
        [[2.0], [0.0 + 0.08255], [0.4], [1.0]],
    ];

    // List of points we saw the target at. These are exactly what we expect
    // for a camera located at 0, 0, 0 (hand-calculated).
    let point_observations = [(325.0, 30.0), (275.0, 30.0)];

    // Initial guess at robot pose. We expect the robot to converge to 0, 0, 0.
    problem.set_initial(robot_x, -0.1);
    problem.set_initial(robot_y, 0.0);
    problem.set_initial(robot_θ, 0.2);

    // field2camera * field2camera⁻¹ = I
    let camera2field = field2camera.solve(VariableMatrix::identity(4));

    // Cost
    let mut cost = Variable::from(0.0);
    for (field2point, (u_observed, v_observed)) in field2points.into_iter().zip(point_observations)
    {
        // camera2point = field2camera⁻¹ * field2point
        // field2camera * camera2point = field2point
        let camera2point = &camera2field * field2point;

        // Point's coordinates in camera frame
        let (x, y, z) = (camera2point[0], camera2point[1], camera2point[2]);

        println!("camera2point = {}, {}, {}", x.value(), y.value(), z.value());

        // Coordinates observed at
        let big_x = x / z;
        let big_y = y / z;

        let u = fx * big_x + cx;
        let v = fy * big_y + cy;

        println!("Expected u {}, saw {u_observed}", u.value());
        println!("Expected v {}, saw {v_observed}", v.value());

        let u_err = u - u_observed;
        let v_err = v - v_observed;

        // Cost function is square of reprojection error
        cost += u_err * u_err + v_err * v_err;
    }

    problem.minimize(cost);

    problem
        .solve(&Options {
            diagnostics: true,
            ..Default::default()
        })
        .unwrap();

    println!("x = {} m", robot_x.value());
    println!("y = {} m", robot_y.value());
    println!("θ = {} rad", robot_θ.value());
}
