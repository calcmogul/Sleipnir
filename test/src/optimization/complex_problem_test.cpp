// Copyright (c) Sleipnir contributors

#include <complex>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Complex problem - AC optimal power flow", "[Complex]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  // Three buses with generators at buses 0 and 1 and a load at bus 2. Every
  // pair of buses is connected by a line with impedance 0.01 + 0.1i p.u., so
  // swapping buses 0 and 1 leaves the problem unchanged.
  constexpr int buses = 3;
  // y = 1/(0.01 + 0.1i) = (0.01 − 0.1i)/0.0101
  const C y{T(0.01 / 0.0101), T(-0.1 / 0.0101)};

  std::vector<Eigen::Triplet<C>> triplets;
  for (auto [i, j] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}}) {
    triplets.emplace_back(i, i, y);
    triplets.emplace_back(j, j, y);
    triplets.emplace_back(i, j, -y);
    triplets.emplace_back(j, i, -y);
  }
  Eigen::SparseMatrix<C> Y{buses, buses};
  Y.setFromTriplets(triplets.begin(), triplets.end());

  const C S_load{T(1), T(0.3)};

  slp::Problem<T> problem;

  auto V = problem.complex_decision_variable(buses);
  auto S_gen = problem.complex_decision_variable(2);
  V.set_value(Eigen::VectorX<T>::Ones(buses));

  // Complex power injected at each bus
  auto S = V.cwise_product((Y * V).conj());

  problem.minimize(S_gen[0].real() * S_gen[0].real() +
                   S_gen[1].real() * S_gen[1].real());

  // Reference angle
  problem.subject_to(V[0].imag() == 0);

  // Power balance
  problem.subject_to(S.segment(0, 2) == S_gen);
  problem.subject_to(S[2] == -S_load);

  // Voltage magnitude limits
  problem.subject_to(V.abs2() >= T(0.95 * 0.95));
  problem.subject_to(V.abs2() <= T(1.05 * 1.05));

  problem.subject_to(S_gen.real() >= 0);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::VectorX<C> v = V.value();
  Eigen::VectorX<C> s_gen = S_gen.value();
  Eigen::VectorX<C> s = v.cwiseProduct((Y * v).conjugate());

  constexpr T tolerance(1e-5);

  // Power balance holds at the solution
  CHECK_THAT(s[0].real(), WithinAbs(s_gen[0].real(), tolerance));
  CHECK_THAT(s[0].imag(), WithinAbs(s_gen[0].imag(), tolerance));
  CHECK_THAT(s[1].real(), WithinAbs(s_gen[1].real(), tolerance));
  CHECK_THAT(s[1].imag(), WithinAbs(s_gen[1].imag(), tolerance));
  CHECK_THAT(s[2].real(), WithinAbs(-S_load.real(), tolerance));
  CHECK_THAT(s[2].imag(), WithinAbs(-S_load.imag(), tolerance));
  CHECK_THAT(v[0].imag(), WithinAbs(T(0), tolerance));

  // The symmetric network splits the load evenly
  CHECK_THAT(s_gen[0].real(), WithinAbs(s_gen[1].real(), tolerance));

  // Line losses are positive and small
  T losses = s_gen[0].real() + s_gen[1].real() - S_load.real();
  CHECK(losses > T(0));
  CHECK(losses < T(0.05));

  // Voltage magnitudes are within their limits
  Eigen::VectorX<T> v_abs2 = V.abs2().value();
  for (int i = 0; i < buses; ++i) {
    CHECK(v_abs2[i] >= T(0.95 * 0.95) - tolerance);
    CHECK(v_abs2[i] <= T(1.05 * 1.05) + tolerance);
  }
}
