// Copyright (c) Sleipnir contributors

// GCC 14 reports a 16-byte SSE packet read from a 12-byte VectorX<float> in a
// vectorized assignment loop branch that never executes for size 3
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wstringop-overread"
#endif

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Rotation3 - Default value and set_value", "[Rotation3]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto q = problem.decision_variable(slp::so3);

  CHECK(q.value().w() == T(1));
  CHECK(q.value().vec().isZero());

  // set_value() normalizes
  q.set_value(Eigen::Quaternion<T>{T(2), T(0), T(0), T(0)});
  CHECK(q.value().w() == T(1));

  q.set_value(Eigen::Quaternion<T>{T(1), T(1), T(1), T(1)});
  CHECK_THAT(q.value().w(), WithinAbs(T(0.5), T(1e-6)));
  CHECK_THAT(q.value().x(), WithinAbs(T(0.5), T(1e-6)));
  CHECK_THAT(q.value().y(), WithinAbs(T(0.5), T(1e-6)));
  CHECK_THAT(q.value().z(), WithinAbs(T(0.5), T(1e-6)));
}

TEMPLATE_TEST_CASE("Rotation3 - Operations match Eigen", "[Rotation3]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto q1 = problem.decision_variable(slp::so3);
  auto q2 = problem.decision_variable(slp::so3);
  auto v = problem.decision_variable(slp::sphere, 3);

  const Eigen::Quaternion<T> q1_value{Eigen::AngleAxis<T>{
      T(0.7), Eigen::Vector3<T>{T(1), T(2), T(-1)}.normalized()}};
  const Eigen::Quaternion<T> q2_value{Eigen::AngleAxis<T>{
      T(-2.1), Eigen::Vector3<T>{T(0.5), T(-1), T(3)}.normalized()}};
  const Eigen::Vector3<T> v_value =
      Eigen::Vector3<T>{T(0.2), T(-0.9), T(0.4)}.normalized();
  const Eigen::Vector3<T> p_value{T(1.5), T(-2), T(0.5)};

  q1.set_value(q1_value);
  q2.set_value(q2_value);
  v.set_value(v_value);

  constexpr T tolerance(1e-5);

  // Composition
  auto q12 = q1 * q2;
  auto q12_expected = q1_value * q2_value;
  CHECK_THAT(q12.value().coeffs(), WithinAbs(q12_expected.coeffs(), tolerance));

  // Inverse
  auto q1_inv = q1.inverse();
  CHECK_THAT(q1_inv.value().coeffs(),
             WithinAbs(q1_value.inverse().coeffs(), tolerance));
  CHECK_THAT((q1 * q1_inv).value().coeffs(),
             WithinAbs(Eigen::Quaternion<T>::Identity().coeffs(), tolerance));

  // Rotation matrix
  Eigen::Matrix3<T> R_expected = q1_value.toRotationMatrix();
  CHECK_THAT(q1.to_matrix().value(), WithinAbs(R_expected, tolerance));

  // Rotating a UnitVector returns a UnitVector
  slp::UnitVector<T> rotated_v = v.rotate_by(q1);
  Eigen::VectorX<T> rotated_v_expected = q1_value * v_value;
  CHECK_THAT(rotated_v.value(), WithinAbs(rotated_v_expected, tolerance));

  // Rotating a general vector returns a VariableMatrix
  slp::VariableMatrix<T> rotated_p =
      slp::VariableMatrix<T>{p_value}.rotate_by(q1);
  Eigen::MatrixX<T> rotated_p_expected = q1_value * p_value;
  CHECK_THAT(rotated_p.value(), WithinAbs(rotated_p_expected, tolerance));
}

TEMPLATE_TEST_CASE("Rotation3 - to_vector() value", "[Rotation3]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto q = problem.decision_variable(slp::so3);
  auto r = q.to_vector();

  const Eigen::Vector3<T> axis =
      Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized();

  // Small angles use the series, and large angles use atan2
  auto angle = GENERATE(0.0, 1e-8, 1e-3, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0);

  for (T sign : {T(1), T(-1)}) {
    // q and −q give the same rotation vector
    Eigen::Quaternion<T> q_value{Eigen::AngleAxis<T>{T(angle), axis}};
    q_value.coeffs() *= sign;
    q.set_value(q_value);

    Eigen::MatrixX<T> expected = T(angle) * axis;
    CHECK_THAT(r.value(), WithinAbs(expected, T(1e-5)));
  }
}

TEMPLATE_TEST_CASE("Rotation3 - to_vector() derivatives", "[Rotation3]",
                   SCALAR_TYPES) {
  using T = TestType;

  constexpr T ε = std::numeric_limits<T>::epsilon();

  slp::Problem<T> problem;
  auto q = problem.decision_variable(slp::so3);
  auto r = q.to_vector();
  slp::VariableMatrix<T> wrt{{q.w()}, {q.x()}, {q.y()}, {q.z()}};

  // ‖r‖² = θ_sq, which is smooth near the identity
  slp::Variable<T> θ_sq = r.T() * r;

  slp::Jacobian J{r, wrt};
  slp::Hessian H{θ_sq, wrt};

  // At the identity, r ≈ 2u/w, so ∂r/∂u = 2I and ∂r/∂w = 0, and
  // θ_sq ≈ 4‖u‖²/w², so ∇²θ_sq = diag(0, 8, 8, 8)
  q.set_value(Eigen::Quaternion<T>::Identity());

  Eigen::MatrixX<T> J_expected{{T(0), T(2), T(0), T(0)},
                               {T(0), T(0), T(2), T(0)},
                               {T(0), T(0), T(0), T(2)}};
  CHECK_THAT(Eigen::MatrixX<T>{J.value()},
             WithinAbs(J_expected, rounding_tolerance<T>(1)));

  Eigen::MatrixX<T> H_expected =
      Eigen::Vector4<T>{T(0), T(8), T(8), T(8)}.asDiagonal();
  CHECK_THAT(Eigen::MatrixX<T>{H.value()},
             WithinAbs(H_expected, rounding_tolerance<T>(1)));

  // The series and atan2 branches agree on either side of the switch
  // t = ‖u‖²/w² = ε^(2/17)
  using std::atan;
  using std::pow;
  using std::sqrt;
  const T t_max = pow(ε, T(2) / T(17));
  const T half_angle_max = atan(sqrt(t_max));
  const Eigen::Vector3<T> axis =
      Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized();

  q.set_value(Eigen::Quaternion<T>{
      Eigen::AngleAxis<T>{T(2) * half_angle_max * (T(1) - T(4) * ε), axis}});
  Eigen::MatrixX<T> J_series = J.value();
  Eigen::MatrixX<T> H_series = H.value();

  q.set_value(Eigen::Quaternion<T>{
      Eigen::AngleAxis<T>{T(2) * half_angle_max * (T(1) + T(4) * ε), axis}});
  Eigen::MatrixX<T> J_atan2 = J.value();
  Eigen::MatrixX<T> H_atan2 = H.value();

  CHECK_THAT(J_series, WithinAbs(J_atan2, rounding_tolerance<T>(100)));
  CHECK_THAT(H_series, WithinAbs(H_atan2, rounding_tolerance<T>(1e4)));
  CHECK(J_series.allFinite());
  CHECK(H_series.allFinite());
}

TEMPLATE_TEST_CASE("UnitVector - Values and operators", "[UnitVector]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto n = problem.decision_variable(slp::sphere, 3);
  auto m = problem.decision_variable(slp::sphere, 3);

  CHECK(n.rows() == 3);
  CHECK(n.cols() == 1);
  CHECK(n.value() == Eigen::Vector3<T>{T(1), T(0), T(0)});

  // set_value() normalizes
  n.set_value(Eigen::Vector3<T>{T(0), T(3), T(4)});
  CHECK_THAT(n.value(),
             WithinAbs(Eigen::Vector3<T>{T(0), T(0.6), T(0.8)}, T(1e-6)));
  m.set_value(Eigen::Vector3<T>{T(0), T(0), T(1)});

  constexpr T tolerance(1e-6);
  const Eigen::Vector3<T> n_value{T(0), T(0.6), T(0.8)};
  const Eigen::Vector3<T> m_value{T(0), T(0), T(1)};

  // Element access and iteration
  CHECK_THAT(n.value(1), WithinAbs(T(0.6), tolerance));
  int count = 0;
  for ([[maybe_unused]]
       const auto& elem : n) {
    ++count;
  }
  CHECK(count == 3);

  // Closed operations
  slp::UnitVector<T> neg_n = -n;
  CHECK_THAT(neg_n.value(), WithinAbs(-n_value, tolerance));

  // Operations that drop to VariableMatrix
  const Eigen::Matrix3<T> A{
      {T(1), T(2), T(3)}, {T(4), T(5), T(6)}, {T(7), T(8), T(10)}};
  slp::Variable<T> s{T(2)};

  CHECK_THAT((A * n).value(), WithinAbs(A * n_value, tolerance));
  CHECK_THAT((slp::VariableMatrix<T>{A} * n).value(),
             WithinAbs(A * n_value, tolerance));
  CHECK_THAT((T(2) * n).value(), WithinAbs(T(2) * n_value, tolerance));
  CHECK_THAT((n * T(2)).value(), WithinAbs(T(2) * n_value, tolerance));
  CHECK_THAT((s * n).value(), WithinAbs(T(2) * n_value, tolerance));
  CHECK_THAT((n * s).value(), WithinAbs(T(2) * n_value, tolerance));
  CHECK_THAT((n / T(2)).value(), WithinAbs(n_value / T(2), tolerance));
  CHECK_THAT((n + m).value(), WithinAbs(n_value + m_value, tolerance));
  CHECK_THAT((n - m).value(), WithinAbs(n_value - m_value, tolerance));
  CHECK_THAT((n + m_value).value(), WithinAbs(n_value + m_value, tolerance));
  CHECK_THAT((m_value - n).value(), WithinAbs(m_value - n_value, tolerance));
  CHECK_THAT((n - slp::VariableMatrix<T>{m_value}).value(),
             WithinAbs(n_value - m_value, tolerance));
  CHECK_THAT((n.T() * m).value(0), WithinAbs(n_value.dot(m_value), tolerance));
  CHECK_THAT(n.dot(m).value(), WithinAbs(n_value.dot(m_value), tolerance));
  CHECK_THAT(n.cross(m).value(), WithinAbs(n_value.cross(m_value), tolerance));

  // Implicit conversion
  slp::VariableMatrix<T> n_matrix = n;
  CHECK_THAT(n_matrix.value(), WithinAbs(n_value, tolerance));
}
