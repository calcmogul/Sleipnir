// Copyright (c) Sleipnir contributors

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/twist3.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Problem - Twist between two poses", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Vector3<T> v_true{T(1), T(-2), T(0.5)};
  const Eigen::Vector3<T> ω_true{T(0.3), T(-1.2), T(2)};

  const slp::Pose3<T> start{
      Eigen::Vector3<T>{T(-1), T(0.5), T(2)},
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(1), Eigen::Vector3<T>{T(0), T(1), T(1)}.normalized()}}};
  const auto end = start * slp::Twist3<T>{v_true, ω_true}.exp();

  slp::Problem<T> problem;
  slp::Twist3<T> twist{problem.decision_variable(3),
                       problem.decision_variable(3)};
  problem.subject_to(start * twist.exp() == end);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(twist.linear().value(),
             WithinAbs(Eigen::MatrixX<T>{v_true}, solution_tolerance<T>(1)));
  CHECK_THAT(twist.angular().value(),
             WithinAbs(Eigen::MatrixX<T>{ω_true}, solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - Rotation integration", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // Rotate from the identity to a target in N steps with the least angular
  // velocity effort, which is a constant angular velocity
  constexpr int N = 10;
  constexpr T dt(0.1);
  const Eigen::Vector3<T> rvec{T(0.5), T(2), T(-1)};
  const slp::Rotation3<T> target{rvec};

  slp::Problem<T> problem;
  std::vector<slp::Rotation3<T>> q;
  std::vector<slp::VariableMatrix<T>> ω;
  for (int k = 0; k < N + 1; ++k) {
    q.push_back(problem.decision_variable(slp::so3));
  }
  for (int k = 0; k < N; ++k) {
    ω.push_back(problem.decision_variable(3));
  }

  problem.subject_to(q[0] ==
                     slp::Rotation3<T>{Eigen::Quaternion<T>::Identity()});
  problem.subject_to(q[N] == target);
  slp::Variable<T> J{T(0)};
  for (int k = 0; k < N; ++k) {
    problem.subject_to(q[k + 1] == q[k] * slp::Rotation3<T>{ω[k] * dt});
    J += ω[k].T() * ω[k];
  }
  problem.minimize(J);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  for (int k = 0; k < N; ++k) {
    CHECK_THAT(ω[k].value(), WithinAbs(Eigen::MatrixX<T>{rvec / (T(N) * dt)},
                                       solution_tolerance<T>(10)));
  }
}
