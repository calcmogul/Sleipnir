// Copyright (c) Sleipnir contributors

//! FRC 2022 shooter trajectory optimization.
//!
//! This program finds the initial velocity, pitch, and yaw for a game piece to
//! hit the 2022 FRC game's target that minimizes either time-to-target or
//! initial velocity (see the `minimize()` calls below).
//!
//! This optimization problem formulation uses direct transcription of the
//! flight dynamics, including air resistance.

// Greek identifiers match the math
#![allow(mixed_script_confusables, confusable_idents)]

use std::f64::consts::PI;

use sleipnir::{Options, Problem, VariableMatrix, constrain};

const FIELD_WIDTH: f64 = 8.2296; // 27 ft -> m
const FIELD_LENGTH: f64 = 16.4592; // 54 ft -> m
const TARGET_WRT_FIELD: [f64; 6] = [FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0, 2.64, 0.0, 0.0, 0.0];
const TARGET_RADIUS: f64 = 0.61; // m
const CONE_ANGLE: f64 = PI / 4.0; // rad
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

    let ω = VariableMatrix::from([[0.0], [-2.0], [0.0]]); // rad/s
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
        FIELD_LENGTH / 4.0,
        FIELD_WIDTH / 4.0,
        0.0,
        1.524,
        -1.524,
        0.0,
    ];
    let max_initial_velocity = 10.0; // m/s

    let shooter_wrt_robot = [0.0, 0.0, 1.2, 0.0, 0.0, 0.0];
    let shooter_wrt_field: [f64; 6] =
        std::array::from_fn(|i| robot_wrt_field[i] + shooter_wrt_robot[i]);

    let problem: Problem = Problem::new();

    // Set up duration decision variables
    const N: usize = 50;
    let total_time = problem.decision_variable();
    problem.subject_to(constrain!(total_time >= 0.0));
    problem.set_initial(total_time, 1.0);
    let dt = total_time / N as f64;

    // Ball state in field frame
    //
    //     [x position]
    //     [y position]
    //     [z position]
    // x = [x velocity]
    //     [y velocity]
    //     [z velocity]
    let x = problem.decision_variable_matrix(6, N);
    let p = x.view(0..3, ..);
    let v = x.view(3..6, ..);

    let v0_wrt_shooter = x.view(3..6, 0)
        - [
            [shooter_wrt_field[3]],
            [shooter_wrt_field[4]],
            [shooter_wrt_field[5]],
        ];

    // Position initial guess is linear interpolation between start and end
    // position
    let lerp = |a: f64, b: f64, t: f64| a + t * (b - a);
    for k in 0..N {
        let t = k as f64 / N as f64;
        for i in 0..3 {
            problem.set_initial(
                x[(i, k)],
                lerp(shooter_wrt_field[i], TARGET_WRT_FIELD[i], t),
            );
        }
    }

    // Velocity initial guess is max initial velocity toward target
    let to_target: [f64; 3] = std::array::from_fn(|i| TARGET_WRT_FIELD[i] - shooter_wrt_field[i]);
    let distance = to_target.iter().map(|d| d * d).sum::<f64>().sqrt();
    for k in 0..N {
        for i in 0..3 {
            problem.set_initial(
                v[(i, k)],
                robot_wrt_field[3 + i] + max_initial_velocity * to_target[i] / distance,
            );
        }
    }

    // Shooter initial position
    problem.subject_to(constrain!(
        p.column(0)
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
        (v[(0, 0)] - robot_wrt_field[3]).powf(2.0)
            + (v[(1, 0)] - robot_wrt_field[4]).powf(2.0)
            + (v[(2, 0)] - robot_wrt_field[5]).powf(2.0)
            <= max_initial_velocity * max_initial_velocity
    ));

    // Keep-out region (cylinder with conic bowl)
    //
    //            r_t
    //           |----|
    //   ___ ___  ____
    //    |   |  |   /
    //    |  h|  |  /
    // z_t|   |  |γ/    z
    //    |  ___ |/     ^
    //    |      |      |
    //   ___     |      ⋅-> x
    //
    // (x_c, y_c, z_c) is the tip of the cone, r_t is target radius, z_t is
    // target height.
    //
    //   tan(γ) = r_t/h
    //   h = r_t/tan(γ)
    //
    //   z_c = z_t - h
    //   z_c = z_t - r_t/tan(γ)
    //
    // Constrain ball to be either outside cylinder or inside cone. First, get
    // cylindrical coordinates with respect to cone origin.
    //
    //   r' = hypot(x - x_c, y - y_c)
    //   z' = z - z_c
    //
    // Cylinder:
    //   r' ≥ r_t
    //   (x - x_c)² + (y - y_c)² ≥ r_t²
    //   (x - x_c)² + (y - y_c)² - r_t² ≥ 0
    //
    // Cone:
    //   z' ≥ r'/tan(γ)
    //   z - z_c ≥ r'/tan(γ)
    //   (z - z_c) tan(γ) ≥ r'
    //   (z - z_c)² tan²(γ) ≥ (x - x_c)² + (y - y_c)²
    //   (z - z_c)² tan²(γ) - (x - x_c)² - (y - y_c)² ≥ 0
    //
    // Ball is in keep-out region if both constraints are violated.
    //
    //   max((x - x_c)² + (y - y_c)² - r_t²,
    //       (z - z_c)² tan²(γ) - (x - x_c)² - (y - y_c)²) ≥ 0
    let x_c = TARGET_WRT_FIELD[0];
    let y_c = TARGET_WRT_FIELD[1];
    let z_c = TARGET_WRT_FIELD[2] - TARGET_RADIUS / CONE_ANGLE.tan();
    for k in 0..N {
        let x2 = (p[(0, k)] - x_c).powf(2.0);
        let y2 = (p[(1, k)] - y_c).powf(2.0);
        let z2 = (p[(2, k)] - z_c).powf(2.0);
        let tan = CONE_ANGLE.tan();
        let cylinder = x2 + y2 - TARGET_RADIUS * TARGET_RADIUS;
        let cone = z2 * tan * tan - x2 - y2;
        problem.subject_to(constrain!(cylinder.max(cone) >= 0.0));
    }

    // Dynamics constraints - RK4 integration
    let h = dt;
    for k in 0..N - 1 {
        let x_k = x.column(k).to_matrix();
        let x_k1 = x.column(k + 1);

        let k1 = f(&x_k);
        let k2 = f(&(&x_k + &k1 * (h / 2.0)));
        let k3 = f(&(&x_k + &k2 * (h / 2.0)));
        let k4 = f(&(&x_k + &k3 * h));
        problem.subject_to(constrain!(
            x_k1 == &x_k + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
        ));
    }

    // Require final position is in center of target circle
    problem.subject_to(constrain!(
        p.column(N - 1)
            == [
                [TARGET_WRT_FIELD[0]],
                [TARGET_WRT_FIELD[1]],
                [TARGET_WRT_FIELD[2]]
            ]
    ));

    // Require the final velocity is down
    problem.subject_to(constrain!(v[(2, N - 1)] <= 0.0));

    // Minimize time-to-target
    // problem.minimize(total_time);

    // Minimize initial velocity
    problem.minimize(v0_wrt_shooter.dot(&v0_wrt_shooter));

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
