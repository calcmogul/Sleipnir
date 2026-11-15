// Copyright (c) Sleipnir contributors

//! FRC 2024 shooter trajectory optimization.
//!
//! This program finds the initial velocity, pitch, and yaw for a game piece to
//! hit the 2024 FRC game's target that minimizes either z sensitivity to
//! initial velocity or initial velocity (see the `minimize()` calls below).
//!
//! This optimization problem formulation uses single-shooting on the flight
//! dynamics, including air resistance, to allow minimizing z sensitivity.

// Greek identifiers match the math
#![allow(mixed_script_confusables, confusable_idents)]

use std::f64::consts::PI;

use sleipnir::{Gradient, Options, Problem, VariableMatrix, constrain};

const FIELD_WIDTH: f64 = 8.2296; // 27 ft -> m
const FIELD_LENGTH: f64 = 16.4592; // 54 ft -> m
const TARGET_LOWER_EDGE: f64 = 1.98; // m
const TARGET_UPPER_EDGE: f64 = 2.11; // m
const TARGET_DEPTH: f64 = 0.46; // m
const TARGET_WRT_FIELD: [f64; 6] = [
    FIELD_LENGTH - TARGET_DEPTH / 2.0,
    FIELD_WIDTH - 2.6575,
    (TARGET_UPPER_EDGE + TARGET_LOWER_EDGE) / 2.0,
    0.0,
    0.0,
    0.0,
];
const G: f64 = 9.806; // m/s²

fn cross<'p>(a: &VariableMatrix<'p, f64>, b: &VariableMatrix<'p, f64>) -> VariableMatrix<'p, f64> {
    VariableMatrix::from([
        [a[1] * b[2] - a[2] * b[1]],
        [a[2] * b[0] - a[0] * b[2]],
        [a[0] * b[1] - a[1] * b[0]],
    ])
}

fn f<'p>(x: &VariableMatrix<'p, f64>) -> VariableMatrix<'p, f64> {
    // x' = x'
    // y' = y'
    // z' = z'
    // [x"]   [ 0]
    // [y"] = [ 0] − F_D(v)/m v̂ − F_L(v)/m (ω x v)
    // [z"]   [−g]
    //
    // ρ is the fluid density in kg/m³
    // v is the linear velocity in m/s
    // v̂ is the velocity direction unit vector
    // ω is the angular velocity in rad/s
    // A is the cross-sectional area of a circle in m²
    // m is the object mass in kg
    let ρ = 1.204; // kg/m³
    let v = x.view(3..6, ..).to_matrix();
    let v2 = v.dot(&v);
    let v_norm = v2.sqrt();
    let v_hat = &v / v_norm;

    let ω = VariableMatrix::from([[0.0], [0.0], [2.0]]); // rad/s
    let r = 0.15; // m
    let a = PI * r * r; // m²
    let m = 0.283; // kg

    // Per https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation:
    //   F_D(v) = ½ρ|v|²C_D A
    //   C_D is the drag coefficient (dimensionless)
    let c_d = 0.5;
    let f_d = 0.5 * ρ * v2 * c_d * a;

    // Magnus force:
    //   F_L(v) = ½ρ|v|C_L A
    //   C_L is the lift coefficient (dimensionless)
    let c_l = 0.5;
    let f_l = 0.5 * ρ * v_norm * c_l * a;

    let accel =
        -VariableMatrix::from([[0.0], [0.0], [G]]) - v_hat * (f_d / m) - cross(&v, &ω) * (f_l / m);
    VariableMatrix::block(&[&[&v], &[&accel]])
}

fn main() {
    // Robot initial state
    let robot_wrt_field = [
        0.75 * FIELD_LENGTH,
        FIELD_WIDTH / 3.0,
        0.0,
        1.524,
        -1.524,
        0.0,
    ];
    let max_initial_velocity = 15.0; // m/s

    let shooter_wrt_robot = [0.0, 0.0, 0.6096, 0.0, 0.0, 0.0];
    let shooter_wrt_field: [f64; 6] =
        std::array::from_fn(|i| robot_wrt_field[i] + shooter_wrt_robot[i]);

    let problem: Problem = Problem::new();

    // Set up duration decision variables
    const N: usize = 10;
    let total_time = problem.decision_variable();
    problem.subject_to(constrain!(total_time >= 0.0));
    problem.set_initial(total_time, 1.0);
    let dt = total_time / N as f64;

    // Disc state in field frame
    //
    //     [x position]
    //     [y position]
    //     [z position]
    // x = [x velocity]
    //     [y velocity]
    //     [z velocity]
    let x = problem.decision_variable_matrix(6, 1);

    // Position initial guess is start position
    for i in 0..3 {
        problem.set_initial(x[i], shooter_wrt_field[i]);
    }

    // Velocity initial guess is max initial velocity toward target
    let to_target: [f64; 3] = std::array::from_fn(|i| TARGET_WRT_FIELD[i] - shooter_wrt_field[i]);
    let distance = to_target.iter().map(|d| d * d).sum::<f64>().sqrt();
    for i in 0..3 {
        problem.set_initial(
            x[3 + i],
            robot_wrt_field[3 + i] + max_initial_velocity * to_target[i] / distance,
        );
    }

    let v0_wrt_shooter = x.view(3..6, ..)
        - [
            [shooter_wrt_field[3]],
            [shooter_wrt_field[4]],
            [shooter_wrt_field[5]],
        ];

    // Shooter initial position
    problem.subject_to(constrain!(
        x.view(0..3, ..)
            == [
                [shooter_wrt_field[0]],
                [shooter_wrt_field[1]],
                [shooter_wrt_field[2]]
            ]
    ));

    // Require initial velocity is below max
    //
    //   √(v_x² + v_y² + v_z²) ≤ vₘₐₓ
    //   v_x² + v_y² + v_z² ≤ vₘₐₓ²
    problem.subject_to(constrain!(
        (x[3] - robot_wrt_field[3]).powf(2.0)
            + (x[4] - robot_wrt_field[4]).powf(2.0)
            + (x[5] - robot_wrt_field[5]).powf(2.0)
            <= max_initial_velocity * max_initial_velocity
    ));

    // Dynamics constraints - RK4 integration
    let h = dt;
    let mut x_k = x.clone();
    for _ in 0..N - 1 {
        let k1 = f(&x_k);
        let k2 = f(&(&x_k + &k1 * (h / 2.0)));
        let k3 = f(&(&x_k + &k2 * (h / 2.0)));
        let k4 = f(&(&x_k + &k3 * h));
        x_k += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
    }

    // Require final position is in center of target circle
    problem.subject_to(constrain!(
        x_k.view(0..3, ..)
            == [
                [TARGET_WRT_FIELD[0]],
                [TARGET_WRT_FIELD[1]],
                [TARGET_WRT_FIELD[2]]
            ]
    ));

    // Require the final velocity is up
    problem.subject_to(constrain!(x_k[5] >= 0.0));

    // Minimize sensitivity of vertical position to velocity
    let sensitivity = Gradient::new(x_k[2], x.view(3..6, ..)).get();
    problem.minimize(sensitivity.dot(&sensitivity));

    // Minimize initial velocity
    // problem.minimize(v0_wrt_shooter.dot(&v0_wrt_shooter));

    problem
        .solve(&Options {
            diagnostics: true,
            ..Default::default()
        })
        .unwrap();

    // Initial velocity vector with respect to shooter
    let v0 = v0_wrt_shooter.values();

    let velocity = v0.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("Velocity = {velocity:.3} m/s");

    let pitch = v0[2].atan2(v0[0].hypot(v0[1]));
    println!("Pitch = {:.3}°", pitch.to_degrees());

    let yaw = v0[1].atan2(v0[0]);
    println!("Yaw = {:.3}°", yaw.to_degrees());

    println!("Total time = {:.3} s", total_time.value());
    println!("dt = {:.3} ms", dt.value() * 1e3);
}
