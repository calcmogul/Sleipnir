// Copyright (c) Sleipnir contributors

// Solves the guided fuel-optimal landing diversion (G-FOLD) problem.
//
// The coordinate system is +X up, +Y east, +Z north.
//
// [1] Açıkmeşe et al., "Lossless Convexification of Nonconvex Control Bound and
//     Pointing Constraints of the Soft Landing Optimal Control Problem", 2013.
//     http://www.larsblackmore.com/iee_tcst13.pdf
// [2] Açıkmeşe et al., "Convex Programming Approach to Powered Descent Guidance
//     for Mars Landing", 2007. https://sci-hub.st/10.2514/1.27553

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <numbers>
#include <numeric>
#include <print>
#include <tuple>

#include <Eigen/Core>
#include <sleipnir/autodiff/slice.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <unsupported/Eigen/MatrixFunctions>

struct Solution {
  slp::ExitStatus status;
  Eigen::MatrixXd X_value;
  Eigen::MatrixXd Z_value;
  Eigen::MatrixXd U_value;
  Eigen::MatrixXd σ_value;

  double cost() const { return σ_value.sum(); }

  bool operator<(const Solution& rhs) const {
    if (status != rhs.status) {
      return status == slp::ExitStatus::SUCCESS;
    } else {
      return cost() < rhs.cost();
    }
  }
};

/// Formatter for Solution.
template <>
struct std::formatter<Solution> {
  /// Parses format string.
  ///
  /// @param ctx Format parse context.
  /// @return Format parse context iterator.
  constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

  /// Formats Solution.
  ///
  /// @tparam FmtContext Format context type.
  /// @param sol Solution.
  /// @param ctx Format context.
  /// @return Format context iterator.
  template <typename FmtContext>
  auto format(const Solution& sol, FmtContext& ctx) const {
    if (sol.status == slp::ExitStatus::SUCCESS) {
      return std::format_to(ctx.out(), "feasible, cost = {}", sol.cost());
    } else {
      return std::format_to(ctx.out(), "infeasible");
    }
  }
};

/// Discretizes the given continuous A and B matrices.
///
/// @tparam States Number of states.
/// @tparam Inputs Number of inputs.
/// @param cont_A Continuous system matrix.
/// @param cont_B Continuous input matrix.
/// @param dt Discretization timestep.
/// @param disc_A Storage for discrete system matrix.
/// @param disc_B Storage for discrete input matrix.
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

/// Returns the minimum of the unimodal function f within [a, c].
std::tuple<int, Solution> line_search(std::function<Solution(int)> f, int a,
                                      int c) {
  // https://en.wikipedia.org/wiki/Golden-section_search

  double ϕ_inv = (std::sqrt(5) - 1) / 2;

  int b = std::round(a + ϕ_inv * (c - a));
  auto b_sol = f(b);
  std::println("\tN ∈ [{}, {}] -> N = {}: {}", std::min(a, c), std::max(a, c),
               b, b_sol);

  while (std::abs(c - a) > 1) {
    auto x = std::round(a + ϕ_inv * (b - a));
    auto x_sol = f(x);
    std::println("\tN ∈ [{}, {}] -> N = {}: {}", std::min(a, c), std::max(a, c),
                 x, x_sol);

    if (x_sol < b_sol) {
      b_sol = x_sol;
      c = b;
      b = x;
    } else {
      a = c;
      c = x;
    }
  }
  return {b, b_sol};
}

#ifndef RUNNING_TESTS
int main() {
  using namespace slp::slicing;

  // From section IV of [1]:

  // Initial lander mass including fuel (kg)
  constexpr double m_wet = 2000.0;

  // Initial fuel mass (kg)
  constexpr double m_fuel = 300.0;

  // Lander dry mass (kg)
  constexpr double m_dry = m_wet - m_fuel;

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
  constexpr double dt = 0.5;

  // If ending straight, use different thrust magnitude constraints at end
  constexpr bool END_STRAIGHT = true;

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
  double t_min = m_dry * v_0.norm() / ρ_2;
  constexpr double t_max = m_fuel / (α * ρ_1);

  // Number of control intervals
  //
  // See equation (57) of [2].
  int N_min = std::ceil(t_min / dt);
  int N_max = std::floor(t_max / dt);

  // Set initial solution storage size to N = 1
  Eigen::MatrixXd X_value{6, 2};
  Eigen::MatrixXd U_value{1, 2};
  Eigen::MatrixXd Z_value{3, 1};
  Eigen::MatrixXd σ_value{1, 1};

  auto solve = [&](int N) -> Solution {
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

    // Initial mass
    problem.subject_to(Z[0, 0] == std::log(m_wet));

    // Final position
    problem.subject_to(q[_, N] == q_f);

    // Final velocity
    problem.subject_to(v[_, N] == v_f);

    // Position and velocity initial guesses
    for (int k = 0; k < N + 1; ++k) {
      for (int i = 0; i < 3; ++i) {
        q[i, k].set_value(
            std::lerp(q_0(i, 0), q_f[i], static_cast<double>(k) / N));
        v[i, k].set_value(
            std::lerp(v_0(i, 0), v_f[i], static_cast<double>(k) / N));
      }
    }

    // State, input, and dynamics constraints
    for (int k = 0; k < N + 1; ++k) {
      double t = k * dt;

      auto x_k = X[_, slp::Slice{k, k + 1}];
      auto q_k = X[slp::Slice{_, 3}, slp::Slice{k, k + 1}];
      auto v_k = X[slp::Slice{3, 6}, slp::Slice{k, k + 1}];
      auto z_k = Z[_, slp::Slice{k, k + 1}];

      // Velocity limits
      problem.subject_to(v_k.T() * v_k <= v_max * v_max);

      // Mass initial guess
      double z_min = std::log(m_wet - α * ρ_2 * t);
      double z_max = std::log(m_wet - α * ρ_1 * t);
      double z_estimate = (z_min + z_max) / 2;
      z_k.set_value(z_estimate);

      if (k < N) {
        auto x_k1 = X[_, slp::Slice{k + 1, k + 2}];
        auto z_k1 = Z[_, slp::Slice{k + 1, k + 2}];
        auto u_k = U[_, slp::Slice{k, k + 1}];
        auto σ_k = σ[_, slp::Slice{k, k + 1}];

        // Input initial guess
        //
        //   ρ₁ ≤ |T_c| ≤ ρ₂
        //   ρ₁ ≤ |u| exp(z) ≤ ρ₂
        //   ρ₁/exp(z) ≤ |u| ≤ ρ₂/exp(z)
        double u_min = ρ_1 / std::exp(z_estimate);
        double u_max = ρ_2 / std::exp(z_estimate);
        u_k.set_value(Eigen::Vector3d{{(u_min + u_max) / 2}, {0.0}, {0.0}});

        // Glide slope constraint on all but the final sample, which ensures the
        // trajectory isn't too shallow or goes below the target height
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

        problem.subject_to(σ_k >= 0);

        if (k == N - 1 && END_STRAIGHT) {
          // Union of the following constraints:
          //
          //   uₖ[0] ≥ σₖ   x pointing constraint
          //   uₖ[1] = 0    y pointing constraint
          //   uₖ[2] = 0    z pointing constraint
          //   uₖᵀuₖ ≤ σₖ²  thrust magnitude limit
          problem.subject_to(u_k[0, 0] == σ_k);
          problem.subject_to(u_k[1, 0] == 0);
          problem.subject_to(u_k[2, 0] == 0);
        } else {
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
        }

        // Thrust slack limits
        //
        // See equation (34) of [2].
        double z_0 = std::log(m_wet - α * ρ_2 * t);
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

    return Solution{status, X.value(), Z.value(), U.value(), σ.value()};
  };

  // Find N with minimum fuel use
  std::println("Searching N ∈ [{}, {}] for minimum fuel use", N_min, N_max);
  auto [N, sol] = line_search(solve, N_min, N_max);
  std::println("N = {} {}", N, sol);
}
#endif
