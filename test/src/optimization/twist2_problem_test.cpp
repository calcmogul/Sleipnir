// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/twist2.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Problem - Pose2 integration", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // Drive from the origin to a pose facing backward in N steps of a
  // differential drive robot, whose body-frame velocity has no sideways
  // component, with the least effort
  constexpr int N = 20;
  constexpr T dt(0.1);
  const slp::Pose2<T> target{Eigen::Vector2<T>{T(2), T(1)},
                             T(std::numbers::pi)};

  slp::Problem<T> problem;
  std::vector<slp::Pose2<T>> poses;
  std::vector<slp::Variable<T>> v;
  std::vector<slp::Variable<T>> ω;
  for (int k = 0; k < N + 1; ++k) {
    poses.push_back(problem.decision_variable(slp::se2));
  }
  for (int k = 0; k < N; ++k) {
    v.push_back(problem.decision_variable());
    ω.push_back(problem.decision_variable());
  }

  problem.subject_to(poses[0] ==
                     slp::Pose2<T>{Eigen::Vector2<T>::Zero(), T(0)});
  problem.subject_to(poses[N] == target);
  slp::Variable<T> J{T(0)};
  for (int k = 0; k < N; ++k) {
    slp::Twist2<T> twist{slp::VariableMatrix<T>{{v[k]}, {T(0)}}, ω[k]};
    problem.subject_to(poses[k + 1] == poses[k] * (twist * dt).exp());
    J += v[k] * v[k] + ω[k] * ω[k];
  }
  problem.minimize(J);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  using std::abs;
  CHECK_THAT(abs(poses[N].rotation().value()),
             WithinAbs(T(std::numbers::pi), solution_tolerance<T>(1)));
  CHECK_THAT(poses[N].translation().x().value(),
             WithinAbs(T(2), solution_tolerance<T>(1)));
}
