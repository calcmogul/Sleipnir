// Copyright (c) Sleipnir contributors

//! Solves the guided fuel-optimal landing diversion (G-FOLD) problem.
//!
//! The coordinate system is +X up, +Y east, +Z north. See the C++ example's
//! README.md for the time horizon derivations.
//!
//! [1] Açıkmeşe et al., "Lossless Convexification of Nonconvex Control Bound
//!     and Pointing Constraints of the Soft Landing Optimal Control Problem",
//!     2013. <http://www.larsblackmore.com/iee_tcst13.pdf>
//! [2] Açıkmeşe et al., "Convex Programming Approach to Powered Descent
//!     Guidance for Mars Landing", 2007. <https://sci-hub.st/10.2514/1.27553>

// Greek identifiers match the math
#![allow(
    mixed_script_confusables,
    confusable_idents,
    uncommon_codepoints,
    non_snake_case
)]

use std::cmp::Ordering;
use std::f64::consts::PI;
use std::fmt;

use sleipnir::nalgebra::{Matrix3, SMatrix, Vector3};
use sleipnir::{Options, Problem, SolveError, SolveStatus, VariableMatrix, constrain};

/// Returns S(ω), the cross product matrix of ω, so S(ω)a = ω×a.
///
/// See equation (2) of [1].
fn cross_matrix(ω: &Vector3<f64>) -> Matrix3<f64> {
    //        [ 0  −ω₃  ω₂]
    // S(ω) = [ ω₃  0  −ω₁]
    //        [−ω₂  ω₁  0 ]
    Matrix3::new(0.0, -ω[2], ω[1], ω[2], 0.0, -ω[0], -ω[1], ω[0], 0.0)
}

/// Returns the time horizon upper bound t_max (s) for a flight from (q₀, v₀)
/// to (q_f, v_f) in a planet frame rotating at ω. Any longer flight would need
/// a larger velocity change than the thrust can provide.
///
/// ω must be nonzero.
fn time_horizon_upper_bound(
    q_0: &Vector3<f64>,
    v_0: &Vector3<f64>,
    q_f: &Vector3<f64>,
    v_f: &Vector3<f64>,
    g: &Vector3<f64>,
    ω: &Vector3<f64>,
    Δv_max: f64,
) -> f64 {
    let S = cross_matrix(ω);

    // Integrating the dynamics in an inertial frame removes the planet
    // rotation terms and gives |Δv_u(t_f)| ≤ Δv_max where
    //
    //   Δv_u(t) = R(t)v_I,f − v_I,0 − R_int(t)g
    //
    // and v_I = v + ω×q is the inertial velocity in the planet frame.
    //
    // t_max is the largest t_f that satisfies the inequality, so we maximize
    // t_f subject to it.
    //
    // See equation (11) of README.md.
    let problem: Problem = Problem::new();
    let t_f = problem.decision_variable();

    // R(t) = I + k₁(t)S(ω) + k₂(t)S(ω)²
    // R_int(t) = tI + k₂(t)S(ω) + k₃(t)S(ω)²
    //
    // See equations (8) and (9) of README.md.
    let ω_norm = ω.norm();
    let k_1 = (t_f * ω_norm).sin() / ω_norm;
    let k_2 = (1.0 - (t_f * ω_norm).cos()) / (ω_norm * ω_norm);
    let k_3 = (t_f - k_1) / (ω_norm * ω_norm);

    // See equation (5) of README.md.
    let v_I_0 = v_0 + ω.cross(q_0);
    let v_I_f = v_f + ω.cross(q_f);

    // Δv_u(t_f) = R(t_f)v_I,f − v_I,0 − R_int(t_f)g
    //           = v_I,f − v_I,0 − t_f g
    //             + k₁(t_f)S(ω)v_I,f
    //             + k₂(t_f)(S(ω)²v_I,f − S(ω)g)
    //             − k₃(t_f)S(ω)²g
    //
    // See equation (10) of README.md.
    let vm = |v: Vector3<f64>| VariableMatrix::from(&v);
    let Δv_u =
        vm(v_I_f - v_I_0) - vm(*g) * t_f + vm(S * v_I_f) * k_1 + vm(S * S * v_I_f - S * g) * k_2
            - vm(S * S * g) * k_3;

    problem.maximize(t_f);
    problem.subject_to(constrain!(Δv_u.dot(&Δv_u) <= Δv_max * Δv_max));
    let _ = problem.solve(&Options::default());

    t_f.value()
}

/// Returns the time horizon lower bound t_min (s) for a flight from (q₀, v₀)
/// to q_f with zero final vertical velocity in a planet frame rotating at ω.
/// +X is up. Any shorter flight would need more vertical acceleration than
/// gravity and the thrust can provide.
///
/// g must point down, a_max is the maximum thrust acceleration (m/s²), θ is
/// the thrust pointing limit from vertical (rad), and t_max is the time
/// horizon upper bound (s).
#[allow(clippy::too_many_arguments)]
fn time_horizon_lower_bound(
    q_0: &Vector3<f64>,
    v_0: &Vector3<f64>,
    q_f: &Vector3<f64>,
    g: &Vector3<f64>,
    ω: &Vector3<f64>,
    v_max: f64,
    a_max: f64,
    θ: f64,
    t_max: f64,
) -> f64 {
    // The planet rotation terms add at most a_ω of vertical acceleration in
    // either direction.
    //
    // See equations (24) and (25) of README.md.
    let q_bound = (q_0.norm() + q_f.norm() + v_max * t_max) / 2.0;
    let a_ω = Vector3::x().cross(ω).norm() * (2.0 * v_max + ω.norm() * q_bound);

    // The thrust pointing limit keeps the vertical thrust acceleration within
    // [min(0, cos(θ))a_max, a_max], so the vertical acceleration lies within
    // [−a_fall, a_brake].
    //
    // See equations (22), (26), and (27) of README.md.
    let a_fall = -g[0] - 0.0_f64.min(θ.cos()) * a_max + a_ω;
    let a_brake = a_max + g[0] + a_ω;

    // The fastest descent falls at a_fall up to a peak downward speed s_peak,
    // then brakes at a_brake.
    //
    // See equations (28) and (29) of README.md.
    let Δh = q_0[0] - q_f[0];
    let s_0 = -v_0[0];
    let s_peak = ((2.0 * Δh + s_0 * s_0 / a_fall) / (1.0 / a_fall + 1.0 / a_brake)).sqrt();
    (s_peak - s_0) / a_fall + s_peak / a_brake
}

/// Discretizes the continuous A and B matrices with timestep dt.
fn discretize_ab(
    A_c: &SMatrix<f64, 6, 6>,
    B_c: &SMatrix<f64, 6, 3>,
    dt: f64,
) -> (SMatrix<f64, 6, 6>, SMatrix<f64, 6, 3>) {
    // M = [A_c  B_c]
    //     [ 0    0 ]
    let mut M = SMatrix::<f64, 9, 9>::zeros();
    M.fixed_view_mut::<6, 6>(0, 0).copy_from(A_c);
    M.fixed_view_mut::<6, 3>(0, 6).copy_from(B_c);

    // ϕ = eᴹᵀ = [A_d  B_d]
    //           [ 0    I ]
    let ϕ = (M * dt).exp();

    (
        ϕ.fixed_view::<6, 6>(0, 0).into_owned(),
        ϕ.fixed_view::<6, 3>(0, 6).into_owned(),
    )
}

/// Returns the index and value of the unimodal function f's minimum within
/// [first, last].
fn line_search<T: PartialOrd + Copy + fmt::Display>(
    mut f: impl FnMut(i64) -> T,
    first: i64,
    last: i64,
) -> (i64, T) {
    // https://en.wikipedia.org/wiki/Golden-section_search

    let mut x_3 = first;
    let mut x_1 = last;

    let mut evaluate = |x: i64, x_1: i64, x_3: i64| {
        let f_x = f(x);
        println!(
            "\tN ∈ [{}, {}] -> N = {x}: {f_x}",
            x_1.min(x_3),
            x_1.max(x_3)
        );
        f_x
    };

    if first + 1 == last {
        let f_first = evaluate(first, x_1, x_3);
        let f_last = evaluate(last, x_1, x_3);
        return if f_last < f_first {
            (last, f_last)
        } else {
            (first, f_first)
        };
    }

    let ϕ_inv = (5.0_f64.sqrt() - 1.0) / 2.0;
    let lerp = |a: i64, b: i64, t: f64| (a as f64 + t * (b - a) as f64).round() as i64;

    let mut x_2 = lerp(x_3, x_1, ϕ_inv);
    let mut f_2 = evaluate(x_2, x_1, x_3);

    while (x_3 - x_1).abs() > 2 {
        let x_4 = lerp(x_3, x_2, ϕ_inv);
        let f_4 = evaluate(x_4, x_1, x_3);
        if f_4 < f_2 {
            // Keep [x_2, x_3]
            x_1 = x_2;
            x_2 = x_4;
            f_2 = f_4;
        } else {
            // Keep [x_1, x_4]
            x_3 = x_1;
            x_1 = x_4;
        }
    }

    // If x_1 or x_3 are first or last, they haven't been evaluated yet
    if x_1 == first || x_1 == last {
        let f_1 = evaluate(x_1, x_1, x_3);
        if f_1 < f_2 {
            return (x_1, f_1);
        }
    }
    if x_3 == first || x_3 == last {
        let f_3 = evaluate(x_3, x_1, x_3);
        if f_3 < f_2 {
            return (x_3, f_3);
        }
    }

    (x_2, f_2)
}

/// A solve's outcome, ordered feasible first, then by fuel use.
#[derive(Clone, Copy, Debug)]
struct Solution {
    status: Result<SolveStatus, SolveError>,
    cost: f64,
}

impl Solution {
    fn key(&self) -> (bool, f64) {
        (self.status != Ok(SolveStatus::Success), self.cost)
    }
}

impl PartialEq for Solution {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl PartialOrd for Solution {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.key().partial_cmp(&other.key())
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.status == Ok(SolveStatus::Success) {
            write!(f, "feasible, cost = {}", self.cost)
        } else {
            write!(f, "infeasible")
        }
    }
}

fn main() {
    // From section IV of [1]:

    // Initial lander mass including fuel (kg)
    let m_wet = 2000.0_f64;

    // Initial fuel mass (kg)
    let m_fuel = 300.0;

    // Lander dry mass (kg)
    let m_dry = m_wet - m_fuel;

    // Maximum thrust (N)
    let T_max = 24000.0;

    let ρ_1 = 0.2 * T_max;
    let ρ_2 = 0.8 * T_max;

    // Fuel consumption rate (s/m)
    let α = 5e-4;

    // Initial position (m)
    let q_0 = Vector3::new(2400.0, 450.0, -330.0);

    // Initial velocity (m/s)
    let v_0 = Vector3::new(-10.0, -40.0, 10.0);

    // Final position (m)
    let q_f = Vector3::new(0.0, 0.0, 0.0);

    // Final velocity (m/s)
    let v_f = Vector3::new(0.0, 0.0, 0.0);

    // Gravitational acceleration on Mars (m/s²)
    let g = Vector3::new(-3.71, 0.0, 0.0);

    // Constant angular velocity of planet (rad/s)
    let ω = Vector3::new(2.53e-5, 0.0, 6.62e-5);

    // Thrust pointing limit (rad)
    let θ = 90.0_f64.to_radians();
    assert!((0.0..=PI).contains(&θ));

    // Minimum glide slope
    let γ_gs = 30.0_f64.to_radians();
    assert!((0.0..=PI / 2.0).contains(&γ_gs));

    // Maximum velocity magnitude (m/s)
    let v_max = 90.0;

    // Time between control intervals (s)
    let dt = 0.5;

    // If ending straight, use different thrust magnitude constraints at end
    const END_STRAIGHT: bool = true;

    // See equation (2) of [1].
    let S = cross_matrix(&ω);

    //     [  0        I  ]
    // A = [-S(ω)²  -2S(ω)]
    let mut A = SMatrix::<f64, 6, 6>::zeros();
    A.fixed_view_mut::<3, 3>(0, 3).fill_with_identity();
    A.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-S * S));
    A.fixed_view_mut::<3, 3>(3, 3).copy_from(&(-2.0 * S));

    //     [0]
    // B = [I]
    let mut B = SMatrix::<f64, 6, 3>::zeros();
    B.fixed_view_mut::<3, 3>(3, 0).fill_with_identity();

    let (A_d, B_d) = discretize_ab(&A, &B, dt);

    let solve = |N: i64| -> Solution {
        let N = N as usize;
        let problem: Problem = Problem::new();

        // x = [position, velocity]ᵀ
        let X = problem.decision_variable_matrix(6, N + 1);
        // z = ln(m)
        let Z = problem.decision_variable_matrix(1, N + 1);
        // u = T_c/m
        let U = problem.decision_variable_matrix(3, N);
        // σ = Γ/m
        let σ = problem.decision_variable_matrix(1, N);

        let q = X.view(..3, ..);
        let v = X.view(3..6, ..);

        // Initial position
        problem.subject_to(constrain!(q.column(0) == q_0));

        // Initial velocity
        problem.subject_to(constrain!(v.column(0) == v_0));

        // Initial mass
        problem.subject_to(constrain!(Z[0] == m_wet.ln()));

        // Final position
        problem.subject_to(constrain!(q.column(N) == q_f));

        // Final velocity
        problem.subject_to(constrain!(v.column(N) == v_f));

        // Final mass must be at least dry mass
        problem.subject_to(constrain!(Z[N] >= m_dry.ln()));

        // Position and velocity initial guesses
        let lerp = |a: f64, b: f64, t: f64| a + t * (b - a);
        for k in 0..=N {
            for i in 0..3 {
                let t = k as f64 / N as f64;
                problem.set_initial(q[(i, k)], lerp(q_0[i], q_f[i], t));
                problem.set_initial(v[(i, k)], lerp(v_0[i], v_f[i], t));
            }
        }

        // State, input, and dynamics constraints
        for k in 0..=N {
            let t = k as f64 * dt;

            let v_k = X.view(3..6, k).to_matrix();
            let z_k = Z[k];

            // Velocity limits
            problem.subject_to(constrain!(v_k.dot(&v_k) <= v_max * v_max));

            // Mass initial guess
            let z_min = (m_wet - α * ρ_2 * t).ln();
            let z_max = (m_wet - α * ρ_1 * t).ln();
            let z_estimate = (z_min + z_max) / 2.0;
            problem.set_initial(z_k, z_estimate);

            if k < N {
                let u_k = U.column(k).to_matrix();
                let σ_k = σ[k];

                // Input initial guess
                //
                //   ρ₁ ≤ |T_c| ≤ ρ₂
                //   ρ₁ ≤ |u| exp(z) ≤ ρ₂
                //   ρ₁/exp(z) ≤ |u| ≤ ρ₂/exp(z)
                let u_min = ρ_1 / z_estimate.exp();
                let u_max = ρ_2 / z_estimate.exp();
                problem.set_initial_matrix(&u_k, [[(u_min + u_max) / 2.0], [0.0], [0.0]]);

                // Glide slope constraint on all but the final sample, which
                // ensures the trajectory isn't too shallow or goes below the
                // target height
                //
                // See equation (12) of [1].
                //
                //       [0  1  0]
                //   E = [0  0  1]
                //
                //                      [1/tan(γ_gs)]
                //   c = e₁/tan(γ_gs) = [     0     ]
                //                      [     0     ]
                //
                //   |E(r - r_f)|₂ - cᵀ(r - r_f) ≤ 0                            (12)
                //
                //   hypot((r − r_f)₂, (r − r_f)₃) − (r − r_f)₁/tan(γ_gs) ≤ 0
                //   hypot((r − r_f)₂, (r − r_f)₃) ≤ (r − r_f)₁/tan(γ_gs)
                //   (r − r_f)₁/tan(γ_gs) ≥ hypot((r − r_f)₂, (r − r_f)₃)
                //   (r − r_f)₁²/tan²(γ_gs) ≥ (r − r_f)₂² + (r − r_f)₃²
                //   (r − r_f)₁² ≥ tan²(γ_gs)((r − r_f)₂² + (r − r_f)₃²)
                let tan_γ_gs = γ_gs.tan();
                problem.subject_to(constrain!(
                    (q[(0, k)] - q_f[0]).powf(2.0)
                        >= tan_γ_gs
                            * tan_γ_gs
                            * ((q[(1, k)] - q_f[1]).powf(2.0) + (q[(2, k)] - q_f[2]).powf(2.0))
                ));

                problem.subject_to(constrain!(σ_k >= 0.0));

                if k == N - 1 && END_STRAIGHT {
                    // Union of the following constraints:
                    //
                    //   uₖ[0] ≥ σₖ   x pointing constraint
                    //   uₖ[1] = 0    y pointing constraint
                    //   uₖ[2] = 0    z pointing constraint
                    //   uₖᵀuₖ ≤ σₖ²  thrust magnitude limit
                    problem.subject_to(constrain!(u_k[0] == σ_k));
                    problem.subject_to(constrain!(u_k[1] == 0.0));
                    problem.subject_to(constrain!(u_k[2] == 0.0));
                } else {
                    // Thrust magnitude limit
                    //
                    // See equation (34) of [1].
                    //
                    //   |u|₂ ≤ σ
                    //   u_x² + u_y² + u_z² ≤ σ²
                    problem.subject_to(constrain!(u_k.dot(&u_k) <= σ_k * σ_k));

                    // Thrust pointing limit
                    //
                    // See equation (34) of [1].
                    //
                    //   n̂ᵀu ≥ cos(θ)σ where n̂ = [1  0  0]ᵀ
                    //   [1  0  0]u ≥ cos(θ)σ
                    //   u_x ≥ cos(θ)σ
                    problem.subject_to(constrain!(u_k[0] >= θ.cos() * σ_k));
                }

                // Thrust slack limits
                //
                // See equation (34) of [2].
                let z_0 = (m_wet - α * ρ_2 * t).ln();
                let μ_1 = ρ_1 * (-z_0).exp();
                let μ_2 = ρ_2 * (-z_0).exp();
                let Δz = z_k - z_0;
                let σ_min = μ_1 * (1.0 - Δz + 0.5 * Δz.powf(2.0));
                let σ_max = μ_2 * (1.0 - Δz);
                problem.subject_to(constrain!(σ_min <= σ_k <= σ_max));
                problem.set_initial(σ_k, (σ_min.value() + σ_max.value()) / 2.0);

                // Integrate dynamics
                //
                // See equations (1), (2), and (33) of [1].
                //
                //   ẋ = Ax + B(g + u)
                //   ż = −ασ
                //
                //   xₖ₊₁ = A_d xₖ + B_d(g + uₖ)
                //   zₖ₊₁ = zₖ - αTσₖ
                problem.subject_to(constrain!(
                    X.column(k + 1) == A_d * X.column(k) + B_d * (g + &u_k)
                ));
                problem.subject_to(constrain!(Z[k + 1] == z_k - α * dt * σ_k));
            }
        }

        // Problem 4 from [1]: Minimum fuel
        problem.minimize(σ.sum());
        let status = problem.solve(&Options::default());

        Solution {
            status,
            cost: σ.sum().value(),
        }
    };

    // Time horizon upper bound (s)
    //
    // The fuel caps the thrust's total velocity change at Δv_max.
    let Δv_max = (m_wet / m_dry).ln() / α;
    let t_max = time_horizon_upper_bound(&q_0, &v_0, &q_f, &v_f, &g, &ω, Δv_max);

    // Time horizon lower bound (s)
    //
    // The thrust acceleration is at most ρ₂/m_dry.
    let t_min = time_horizon_lower_bound(&q_0, &v_0, &q_f, &g, &ω, v_max, ρ_2 / m_dry, θ, t_max);

    // Number of control intervals
    //
    // See equation (57) of [2].
    let N_min = (t_min / dt).ceil() as i64;
    let N_max = (t_max / dt).floor() as i64;

    // Find N with minimum fuel use
    println!("Searching N ∈ [{N_min}, {N_max}] for minimum fuel use");
    let (N, solution) = line_search(solve, N_min, N_max);
    println!("N = {N}: {solution}");
}
