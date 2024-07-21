// Copyright (c) Sleipnir contributors

// Solves the guided fuel-optimal landing diversion (G-FOLD) problem using
// Sleipnir.
//
// The coordinate system is +X up, +Y east, +Z north.
//
// [1] Açıkmeşe et al., "Lossless Convexification of Nonconvex Control Bound and
//     Pointing Constraints of the Soft Landing Optimal Control Problem", 2013.
//     http://www.larsblackmore.com/iee_tcst13.pdf
// [2] Açıkmeşe et al., "Convex Programming Approach to Powered Descent Guidance
//     for Mars Landing", 2007. https://sci-hub.st/10.2514/1.27553

#include <cmath>
#include <numbers>
#include <numeric>
#include <print>

#include <sleipnir/optimization/problem.hpp>
#include <unsupported/Eigen/MatrixFunctions>

/// Discretizes the given continuous A and B matrices.
///
/// @tparam States Number of states.
/// @tparam Inputs Number of inputs.
/// @param contA Continuous system matrix.
/// @param contB Continuous input matrix.
/// @param dt Discretization timestep.
/// @param discA Storage for discrete system matrix.
/// @param discB Storage for discrete input matrix.
template <int States, int Inputs>
void discretize_ab(const Eigen::Matrix<double, States, States>& cont_A,
                   const Eigen::Matrix<double, States, Inputs>& cont_B,
                   double dt, Eigen::Matrix<double, States, States>* disc_A,
                   Eigen::Matrix<double, States, Inputs>* disc_B) {
  // M = [A  B]
  //     [0  0]
  Eigen::Matrix<double, States + Inputs, States + Inputs> M;
  M.template block<States, States>(0, 0) = cont_A;
  M.template block<States, Inputs>(0, States) = cont_B;
  M.template block<Inputs, States + Inputs>(States, 0).setZero();

  // ϕ = eᴹᵀ = [A_d  B_d]
  //           [ 0    I ]
  Eigen::Matrix<double, States + Inputs, States + Inputs> phi = (M * dt).exp();

  *disc_A = phi.template block<States, States>(0, 0);
  *disc_B = phi.template block<States, Inputs>(0, States);
}

#ifndef RUNNING_TESTS
int main() {
  using namespace slp::slicing;

  // From section IV of [1]:

  // Initial mass (kg)
  constexpr double m_0 = 2000.0;

  // Final mass (kg)
  constexpr double m_f = 300.0;

  // Maximum thrust (N)
  constexpr double T_max = 24000;

  constexpr double ρ_1 = 0.2 * T_max;
  constexpr double ρ_2 = 0.8 * T_max;

  // Fuel consumption rate (s/m)
  constexpr double α = 5e-4;
  static_assert(α > 0);

  // Initial position (m)
  constexpr Eigen::Vector3d q_0{{2400.0}, {450.0}, {-330.0}};

  // Initial velocity (m/s)
  constexpr Eigen::Vector3d v_0{{-10.0}, {-40.0}, {10.0}};

  // Final position (m)
  constexpr Eigen::Vector3d q_f{{0.0}, {0.0}, {0.0}};

  // Final velocity (m/s)
  constexpr Eigen::Vector3d v_f{{0.0}, {0.0}, {0.0}};

  // Gravitational acceleration on Mars (m/s²)
  constexpr Eigen::Vector3d g{{-3.71}, {0.0}, {0.0}};

  // Constant angular velocity of planet (rad/s)
  constexpr Eigen::Vector3d ω{{2.53e-5}, {0.0}, {6.62e-5}};

  // Thrust pointing limit (rad)
  constexpr double θ = 90.0 * std::numbers::pi / 180.0;
  static_assert(θ >= 0.0 && θ <= std::numbers::pi / 2);

  // Minimum glide slope
  constexpr double γ_gs = 30.0 * std::numbers::pi / 180.0;
  static_assert(γ_gs >= 0.0 && γ_gs <= std::numbers::pi / 2);

  // Maximum velocity magnitude (m/s)
  constexpr double v_max = 90.0;

  // Time between control intervals (s)
  constexpr double dt = 1.0;

  // See equation (2) of [1].

  //     [0   -ω₃  ω₂]
  // S = [ω₃   0  −ω₁]
  //     [−ω₂  ω₁  0 ]
  constexpr double ω_1 = ω[0];
  constexpr double ω_2 = ω[1];
  constexpr double ω_3 = ω[2];
  constexpr Eigen::Matrix3d S{
      {0.0, -ω_3, ω_2}, {ω_3, 0.0, -ω_1}, {-ω_2, ω_1, 0.0}};

  //     [  0        I  ]
  // A = [-S(ω)²  -2S(ω)]
  Eigen::Matrix<double, 6, 6> A;
  A.block<3, 3>(0, 0).setZero();
  A.block<3, 3>(0, 3).setIdentity();
  A.block<3, 3>(3, 0) = -S * S;
  A.block<3, 3>(3, 3) = -2 * S;

  //     [0]
  // B = [I]
  Eigen::Matrix<double, 6, 3> B;
  B.block<3, 3>(0, 0).setZero();
  B.block<3, 3>(3, 0).setIdentity();

  Eigen::Matrix<double, 6, 6> A_d;
  Eigen::Matrix<double, 6, 3> B_d;
  discretize_ab<6, 3>(A, B, dt, &A_d, &B_d);

  // Time horizon bounds (s)
  //
  // See equation (55) of [2].
  double t_min = (m_0 - m_f) * v_0.norm() / ρ_2;
  constexpr double t_max = m_f / (α * ρ_1);

  // Number of control intervals
  //
  // See equation (57) of [2].
  int N_min = std::ceil(t_min / dt);
  int N_max = std::floor(t_max / dt);

  // Bisect to find minimum feasible N
  std::println("Searching N ∈ [{}, {}] for smallest feasible N", N_min, N_max);
  Eigen::MatrixXd X_value = Eigen::MatrixXd::Zero(6, 2);
  Eigen::MatrixXd U_value = Eigen::MatrixXd::Zero(1, 2);
  Eigen::MatrixXd Z_value = Eigen::MatrixXd::Zero(3, 1);
  bool found = false;
  while (!found) {
    int N = N_min + (N_max - N_min) / 2;

    std::print("Trying N = {} from [{}, {}]...", N, N_min, N_max);

    slp::Problem<double> problem;

    // x = [position, velocity]ᵀ
    auto X = problem.decision_variable(6, N + 1);
    // z = ln(m)
    auto Z = problem.decision_variable(1, N + 1);
    // u = T_c/m
    auto U = problem.decision_variable(3, N);
    // σ = Γ/m
    auto σ = problem.decision_variable(1, N);

    auto q = X[slp::Slice{_, 3}, _];
    auto v = X[slp::Slice{3, 6}, _];

    // Initial position
    problem.subject_to(q[_, slp::Slice{_, 1}] == q_0);

    // Initial velocity
    problem.subject_to(v[_, slp::Slice{_, 1}] == v_0);

    // Initial ln(mass)
    problem.subject_to(Z[0] == std::log(m_0));

    // Final x position
    problem.subject_to(q[0, N] == q_f[0]);

    // Final velocity
    problem.subject_to(v[_, N] == v_f);

    // Position and velocity initial guesses
    for (int k = 0; k < N + 1; ++k) {
      for (int i = 0; i < 3; ++i) {
        q[i, k].set_value(std::lerp(q_0(i, 0), q_f[i], k / N));
        v[i, k].set_value(std::lerp(v_0(i, 0), v_f[i], k / N));
      }
    }

    // End straight
    problem.subject_to(U[0, N - 1] >= σ[-1]);
    problem.subject_to(U[1, N - 1] == 0);
    problem.subject_to(U[2, N - 1] == 0);

    // State, input, and dynamics constraints
    for (int k = 0; k < N + 1; ++k) {
      double t = k * dt;

      [[maybe_unused]]
      auto x_k = X[_, slp::Slice{k, k + 1}];
      auto q_k = X[slp::Slice{_, 3}, slp::Slice{k, k + 1}];
      auto v_k = X[slp::Slice{3, 6}, slp::Slice{k, k + 1}];
      auto z_k = Z[_, slp::Slice{k, k + 1}];

      // Mass limits
      double z_min = std::log(m_0 - α * ρ_2 * t);
      double z_max = std::log(m_0 - α * ρ_1 * t);
      problem.subject_to(slp::bounds(z_min, z_k, z_max));
      double z_estimate = (z_min + z_max) / 2;
      z_k.set_value(z_estimate);

      // Glide slope constraint, which ensure the trajectory isn't too shallow
      // or goes below the target height
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
      problem.subject_to(
          slp::pow(q_k[0] - q_f[0], 2) >=
          std::tan(γ_gs) * std::tan(γ_gs) *
              (slp::pow(q_k[1] - q_f[1], 2) + slp::pow(q_k[2] - q_f[2], 2)));

      // Velocity limits
      problem.subject_to(v_k.T() * v_k <= v_max * v_max);

      if (k < N) {
        auto x_k1 = X[_, slp::Slice{k + 1, k + 2}];
        auto z_k1 = Z[_, slp::Slice{k + 1, k + 2}];
        auto u_k = U[_, slp::Slice{k, k + 1}];
        auto σ_k = σ[_, slp::Slice{k, k + 1}];

        problem.subject_to(σ_k >= 0);

        // Input initial guess
        //
        //   ρ₁ ≤ |T_c| ≤ ρ₂
        //   ρ₁ ≤ |u| exp(z) ≤ ρ₂
        //   ρ₁/exp(z) ≤ |u| ≤ ρ₂/exp(z)
        double u_min = ρ_1 / std::exp(z_estimate);
        double u_max = ρ_2 / std::exp(z_estimate);
        u_k.set_value(Eigen::Vector3d{{(u_min + u_max) / 2}, {0.0}, {0.0}});

        // Thrust magnitude limit
        //
        // See equation (34) of [1].
        //
        //   |u|₂ ≤ σ
        //   u_x² + u_y² + u_z² ≤ σ²
        problem.subject_to(u_k.T() * u_k <= σ_k * σ_k);

        // Thrust pointing limit
        //
        // See equation (34) of [1].
        //
        //   n̂ᵀu ≥ cos(θ)σ where n̂ = [1  0  0]ᵀ
        //   [1  0  0]u ≥ cos(θ)σ
        //   u_x ≥ cos(θ)σ
        problem.subject_to(u_k[0] >= std::cos(θ) * σ_k);

        // Thrust slack limits
        //
        // See equation (34) of [2].
        double z_0 = std::log(m_0 - α * ρ_2 * t);
        double μ_1 = ρ_1 * std::exp(-z_0);
        double μ_2 = ρ_2 * std::exp(-z_0);
        auto σ_min =
            μ_1 * (1 - (z_k[0] - z_0) + 0.5 * slp::pow(z_k[0] - z_0, 2));
        auto σ_max = μ_2 * (1 - (z_k[0] - z_0));
        problem.subject_to(slp::bounds(σ_min, σ_k, σ_max));
        σ_k.set_value((σ_min.value() + σ_max.value()) / 2);

        // Integrate dynamics
        //
        // See equation (2) of [1].
        //
        //   ẋ = Ax + B(g + u)
        //   ż = −ασ
        //
        //   xₖ₊₁ = A_d xₖ + B_d(g + uₖ)
        //   zₖ₊₁ = zₖ - αTσₖ
        problem.subject_to(x_k1 == A_d * x_k + B_d * (g + u_k));
        problem.subject_to(z_k1 == z_k - α * dt * σ_k);
      }
    }

    // Problem 4 from [1]: Minimum fuel
    problem.minimize(std::accumulate(σ.begin(), σ.end(), slp::Variable{0.0}));
    auto status = problem.solve();

    if (status == slp::ExitStatus::SUCCESS) {
      std::println(" feasible");

      X_value = X.value();
      U_value = U.value();
      Z_value = Z.value();

      // Problem is feasible, so try a smaller N
      if (N_min < N) {
        N_max = N - 1;
      } else {
        std::println("Smallest feasible N = {}", N);
        found = true;
      }
    } else {
      std::println(" infeasible");

      // Problem is infeasible, so try a larger N
      if (N_min < N_max) {
        N_min = N + 1;
      } else {
        std::println("Smallest feasible N = {}", N + 1);
        found = true;
      }
    }
  }
}
#endif
