// Copyright (c) Sleipnir contributors

// GCC reports the fixed-size 3x3 SelfAdjointEigenSolver's eigenvalues as
// maybe-uninitialized after construction, which computes them
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

namespace {

template <typename T>
Eigen::Matrix3<T> rayleigh_matrix() {
  // Symmetric with distinct eigenvalues
  return Eigen::Matrix3<T>{
      {T(2), T(-1), T(0.5)}, {T(-1), T(3), T(0.25)}, {T(0.5), T(0.25), T(1)}};
}

/// Adds an iteration callback that checks the manifold blocks of the iterate
/// stay unit-length.
template <typename T>
void check_iterates_on_manifold(slp::Problem<T>& problem, int offset,
                                int size) {
  problem.add_callback([=](const slp::IterationInfo<T>& info) {
    CHECK_THAT(info.x.segment(offset, size).norm(),
               WithinAbs(T(1), rounding_tolerance<T>(10)));
  });
}

}  // namespace

TEMPLATE_TEST_CASE("Problem - Sphere Rayleigh quotient", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix3<T> A = rayleigh_matrix<T>();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigensolver{A};
  const T λ_min = eigensolver.eigenvalues()[0];
  const Eigen::Vector3<T> v_min = eigensolver.eigenvectors().col(0);

  auto x0 =
      GENERATE(Eigen::Vector3d{1, 0, 0}, Eigen::Vector3d{-1, 0.2, 0.1},
               Eigen::Vector3d{0.1, 1, -0.3}, Eigen::Vector3d{0.5, -0.5, 0.7});

  slp::Problem<T> problem;
  auto n = problem.decision_variable(slp::sphere, 3);
  n.set_value(x0.template cast<T>().eval());

  problem.minimize(n.T() * A * n);
  check_iterates_on_manifold(problem, 0, 3);

  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Vector3<T> n_value = n.value();
  CHECK_THAT(n_value.norm(), WithinAbs(T(1), rounding_tolerance<T>(10)));
  CHECK_THAT(n_value.dot(A * n_value),
             WithinAbs(λ_min, solution_tolerance<T>(1)));
  using std::abs;
  CHECK_THAT(abs(n_value.dot(v_min)),
             WithinAbs(T(1), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE(
    "Problem - Sphere with constraints matches unit-norm "
    "equality constraint",
    "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix3<T> A = rayleigh_matrix<T>();
  const Eigen::Vector3<T> x0 = Eigen::Vector3<T>{T(1), T(1), T(1)}.normalized();

  // Equality constraint (SQP) or inequality constraint (IPM)
  auto use_inequality = GENERATE(false, true);

  auto add_constraint = [&](slp::Problem<T>& problem, const auto& n) {
    if (use_inequality) {
      problem.subject_to(n[2] >= T(0.6));
    } else {
      problem.subject_to(n[0] == T(0.5));
    }
  };

  // Manifold formulation
  slp::Problem<T> manifold_problem;
  auto n = manifold_problem.decision_variable(slp::sphere, 3);
  n.set_value(x0);
  manifold_problem.minimize(n.T() * A * n);
  add_constraint(manifold_problem, n);
  check_iterates_on_manifold(manifold_problem, 0, 3);

  CHECK(manifold_problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::SUCCESS);

  // Unit-norm equality constraint formulation
  slp::Problem<T> constrained_problem;
  auto m = constrained_problem.decision_variable(3);
  m.set_value(x0);
  constrained_problem.minimize(m.T() * A * m);
  constrained_problem.subject_to(m.T() * m == T(1));
  add_constraint(constrained_problem, m);

  CHECK(constrained_problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::SUCCESS);

  CHECK_THAT(n.value(), WithinAbs(m.value(), solution_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Problem - Wahba's problem", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Quaternion<T> q_true{Eigen::AngleAxis<T>{
      T(2.5), Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized()}};

  const std::vector<Eigen::Vector3<T>> body_vectors{
      {T(1), T(0), T(0)}, {T(0), T(1), T(0)}, {T(0.5), T(0.5), T(0.7)}};

  // Initial guesses from the identity to nearly opposite the solution
  auto initial_angle = GENERATE(0.0, 1.0, 3.0);

  slp::Problem<T> problem;
  auto q = problem.decision_variable(slp::so3);
  q.set_value(Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
      T(initial_angle), Eigen::Vector3<T>{T(0.3), T(1), T(0)}.normalized()}});

  slp::Variable<T> cost{T(0)};
  for (const auto& v : body_vectors) {
    Eigen::Vector3<T> w = q_true * v;
    auto error = slp::VariableMatrix<T>{v}.rotate_by(q) - w;
    cost += error.T() * error;
  }
  problem.minimize(cost);
  check_iterates_on_manifold(problem, 0, 4);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // q and −q are the same rotation
  using std::abs;
  CHECK_THAT(abs(q.value().dot(q_true)),
             WithinAbs(T(1), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - Rotation averaging", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  const std::vector<Eigen::Quaternion<T>> rotations{
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(0.3), Eigen::Vector3<T>{T(1), T(0), T(0)}.normalized()}},
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(0.5), Eigen::Vector3<T>{T(0), T(1), T(0.2)}.normalized()}},
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(-0.4), Eigen::Vector3<T>{T(0.3), T(0.3), T(1)}.normalized()}},
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(0.2), Eigen::Vector3<T>{T(-1), T(0.5), T(0.5)}.normalized()}}};

  slp::Problem<T> problem;
  auto q = problem.decision_variable(slp::so3);

  std::vector<slp::VariableMatrix<T>> residuals;
  slp::Variable<T> cost{T(0)};
  for (const auto& rotation : rotations) {
    auto r = (slp::Rotation3<T>{rotation}.inverse() * q).to_vector();
    cost += r.T() * r;
    residuals.push_back(r);
  }
  problem.minimize(cost);
  check_iterates_on_manifold(problem, 0, 4);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // The minimizer of Σᵢ ‖log(qᵢ⁻¹q)‖² is the Karcher mean, where
  // Σᵢ log(qᵢ⁻¹q) = 0
  Eigen::Vector3<T> sum = Eigen::Vector3<T>::Zero();
  for (auto& r : residuals) {
    sum += Eigen::Vector3<T>{r.value()};
  }
  CHECK_THAT(sum,
             WithinAbs(Eigen::Vector3<T>::Zero(), solution_tolerance<T>(10)));
}
TEMPLATE_TEST_CASE("Problem - 2D point set registration", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  using std::cos;
  using std::sin;

  // The true angle is near π, and the initial guess is on the other side of
  // ±π, where an angle decision variable would have to wrap
  const T θ_true(2.8);
  const Eigen::Matrix2<T> R_true{{cos(θ_true), -sin(θ_true)},
                                 {sin(θ_true), cos(θ_true)}};
  const Eigen::Vector2<T> t_true{T(1), T(-0.5)};

  const std::vector<Eigen::Vector2<T>> points{
      {T(1), T(0)}, {T(0), T(2)}, {T(-1), T(-1)}, {T(2), T(1)}};

  slp::Problem<T> problem;
  auto z = problem.decision_variable(slp::so2);
  auto t = problem.decision_variable(2);
  z.set_value(T(-2.8));

  slp::Variable<T> cost{T(0)};
  for (const auto& p : points) {
    Eigen::Vector2<T> measured = R_true * p + t_true;
    auto error = slp::VariableMatrix<T>{p}.rotate_by(z) + t - measured;
    cost += error.T() * error;
  }
  problem.minimize(cost);
  check_iterates_on_manifold(problem, 0, 2);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(z.value(), WithinAbs(θ_true, solution_tolerance<T>(1)));
  CHECK_THAT(t.value(0), WithinAbs(t_true[0], solution_tolerance<T>(1)));
  CHECK_THAT(t.value(1), WithinAbs(t_true[1], solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - SE(2) point set registration",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const slp::Pose2<T> pose_true{Eigen::Vector2<T>{T(1), T(-0.5)}, T(2.8)};

  const std::vector<Eigen::Vector2<T>> points{
      {T(1), T(0)}, {T(0), T(2)}, {T(-1), T(-1)}, {T(2), T(1)}};

  slp::Problem<T> problem;
  auto pose = problem.decision_variable(slp::se2);
  pose.rotation().set_value(T(-2.8));

  slp::Variable<T> cost{T(0)};
  for (const auto& p : points) {
    const slp::Translation2<T> measured{
        slp::Translation2<T>{p}.transform_by(pose_true).value()};
    cost +=
        slp::Translation2<T>{p}.transform_by(pose).squared_distance(measured);
  }
  problem.minimize(cost);
  check_iterates_on_manifold(problem, 2, 2);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(pose.rotation().value(),
             WithinAbs(T(2.8), solution_tolerance<T>(1)));
  CHECK_THAT(pose.translation().x().value(),
             WithinAbs(T(1), solution_tolerance<T>(1)));
  CHECK_THAT(pose.translation().y().value(),
             WithinAbs(T(-0.5), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - SE(3) point set registration",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Quaternion<T> q_true{Eigen::AngleAxis<T>{
      T(2.5), Eigen::Vector3<T>{T(0.2), T(1), T(-0.4)}.normalized()}};
  const Eigen::Vector3<T> t_true{T(1), T(-0.5), T(2)};

  const std::vector<Eigen::Vector3<T>> points{{T(1), T(0), T(0)},
                                              {T(0), T(2), T(0)},
                                              {T(0), T(0), T(-1)},
                                              {T(1), T(1), T(1)}};

  slp::Problem<T> problem;
  auto pose = problem.decision_variable(slp::se3);

  // The measurements are of the points in the pose's frame
  slp::Variable<T> cost{T(0)};
  for (const auto& p : points) {
    Eigen::Vector3<T> measured = q_true.inverse() * (p - t_true);
    cost += slp::Translation3<T>{p}
                .transform_by(pose.inverse())
                .squared_distance(slp::Translation3<T>{measured});
  }
  problem.minimize(cost);
  check_iterates_on_manifold(problem, 3, 4);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Vector3<T> t = pose.translation().value();
  Eigen::Quaternion<T> q = pose.rotation().value();
  for (int i = 0; i < 3; ++i) {
    CHECK_THAT(t[i], WithinAbs(t_true[i], solution_tolerance<T>(1)));
  }
  using std::abs;
  CHECK_THAT(abs(q.dot(q_true)), WithinAbs(T(1), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - Pose estimation with inequality constraints",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Quaternion<T> q_true{Eigen::AngleAxis<T>{
      T(1.2), Eigen::Vector3<T>{T(0.2), T(1), T(-0.4)}.normalized()}};
  const Eigen::Vector3<T> t_true{T(1), T(-0.5), T(2)};

  const std::vector<Eigen::Vector3<T>> points{{T(1), T(0), T(0)},
                                              {T(0), T(2), T(0)},
                                              {T(0), T(0), T(-1)},
                                              {T(1), T(1), T(1)}};

  slp::Problem<T> problem;
  auto q = problem.decision_variable(slp::so3);
  auto t = problem.decision_variable(3);

  slp::Variable<T> cost{T(0)};
  for (const auto& p : points) {
    Eigen::Vector3<T> measured = q_true * p + t_true;
    auto error = slp::VariableMatrix<T>{p}.rotate_by(q) + t - measured;
    cost += error.T() * error;
  }
  problem.minimize(cost);

  // The translation bound is active, so the solution moves off the
  // unconstrained optimum
  problem.subject_to(t[2] <= T(1.5));
  problem.subject_to(t[0] >= T(-5));
  check_iterates_on_manifold(problem, 0, 4);

  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(t.value(2), WithinAbs(T(1.5), solution_tolerance<T>(1)));
  CHECK_THAT(q.value().norm(), WithinAbs(T(1), rounding_tolerance<T>(10)));
}
