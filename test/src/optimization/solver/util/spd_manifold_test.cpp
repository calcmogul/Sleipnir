// Copyright (c) Sleipnir contributors

#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/optimization/solver/util/manifold.hpp>
#include <sleipnir/optimization/solver/util/manifold_type.hpp>
#include <sleipnir/optimization/solver/util/spd_manifold.hpp>
#include <sleipnir/optimization/solver/util/sphere_manifold.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

namespace {

// Euclidean, 3×3 SPD, Euclidean
template <typename T>
slp::Manifold<T> make_spd_manifold() {
  return slp::Manifold<T>{
      {slp::ManifoldBlock<T>{slp::spd_manifold<T>, 1, 3, 3}}};
}

template <typename T>
Eigen::VectorX<T> make_spd_point() {
  // Lower triangle in row-major order of
  //
  //   [ 2   −0.5  0.3]
  //   [−0.5  1.5  0.2]
  //   [ 0.3  0.2  0.8]
  return Eigen::VectorX<T>{
      {T(0.7), T(2), T(-0.5), T(1.5), T(0.3), T(0.2), T(0.8), T(-1.2)}};
}

/// Returns the symmetric matrix whose lower triangle in row-major order is x.
template <typename T>
Eigen::Matrix3<T> unpack3(const Eigen::VectorX<T>& x) {
  return Eigen::Matrix3<T>{
      {x[0], x[1], x[3]}, {x[1], x[2], x[4]}, {x[3], x[4], x[5]}};
}

}  // namespace

TEMPLATE_TEST_CASE("Manifold - SPD dimensions", "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  // The SPD matrices are open in the symmetric matrices, so no dimension is
  // lost
  CHECK(make_spd_manifold<T>().tangent_dimension(8) == 8);
}

TEMPLATE_TEST_CASE("Manifold - SPD projection", "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  auto manifold = make_spd_manifold<T>();

  // An SPD point is unchanged
  auto x = make_spd_point<T>();
  Eigen::VectorX<T> projected = x;
  manifold.project(projected);
  CHECK(projected == x);

  // Zero goes to the identity
  Eigen::VectorX<T> zero = Eigen::VectorX<T>::Zero(8);
  manifold.project(zero);
  CHECK(unpack3<T>(zero.segment(1, 6)) == Eigen::Matrix3<T>::Identity());
  CHECK(zero[0] == T(0));
  CHECK(zero[7] == T(0));

  // An indefinite matrix gets its negative eigenvalue raised
  Eigen::VectorX<T> indefinite{
      {T(0), T(1), T(0), T(-1), T(0), T(0), T(2), T(0)}};
  manifold.project(indefinite);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigensolver{
      unpack3<T>(indefinite.segment(1, 6))};
  CHECK(eigensolver.eigenvalues().minCoeff() > T(0));
  CHECK_THAT(eigensolver.eigenvalues().maxCoeff(),
             WithinAbs(T(2), rounding_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Manifold - SPD projection keeps ill-conditioned matrices",
                   "[Manifold]", SCALAR_TYPES) {
  using T = TestType;
  using std::pow;
  using std::sqrt;

  constexpr T ε = std::numeric_limits<T>::epsilon();

  auto manifold = make_spd_manifold<T>();

  // A rotation, so the matrices below aren't diagonal
  Eigen::Matrix3<T> V =
      Eigen::AngleAxis<T>{T(0.7),
                          Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized()}
          .toRotationMatrix();

  auto pack = [&](const Eigen::Vector3<T>& λ) {
    Eigen::Matrix3<T> P = V * λ.asDiagonal() * V.transpose();
    P = (P + P.transpose()) / T(2);
    return Eigen::VectorX<T>{
        {T(0), P(0, 0), P(1, 0), P(1, 1), P(2, 0), P(2, 1), P(2, 2), T(0)}};
  };

  // Condition number 1/ε^(3/4) is above 1/√ε but well below 1/(nε), so the
  // matrix is positive definite to working precision and is unchanged
  Eigen::VectorX<T> ill_conditioned = pack({T(1), T(0.5), pow(ε, T(0.75))});
  Eigen::VectorX<T> projected = ill_conditioned;
  manifold.project(projected);
  CHECK(projected == ill_conditioned);

  // A numerically zero eigenvalue is raised to √ε times the largest, and the
  // others are unchanged
  Eigen::VectorX<T> singular = pack({T(1), T(0.5), T(0)});
  manifold.project(singular);
  Eigen::Vector3<T> λ =
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>>{
          unpack3<T>(singular.segment(1, 6))}
          .eigenvalues();
  CHECK_THAT(λ[0], WithinAbs(sqrt(ε), rounding_tolerance<T>(10)));
  CHECK_THAT(λ[1], WithinAbs(T(0.5), rounding_tolerance<T>(10)));
  CHECK_THAT(λ[2], WithinAbs(T(1), rounding_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Manifold - SPD retraction stays positive definite",
                   "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  auto manifold = make_spd_manifold<T>();
  auto x = make_spd_point<T>();

  // A step of −10 P would make P + ξ negative definite
  Eigen::VectorX<T> p = Eigen::VectorX<T>::Zero(8);
  p.segment(1, 6) = T(-10) * x.segment(1, 6);
  p[0] = T(1);

  for (T α : {T(0.1), T(1), T(10)}) {
    Eigen::VectorX<T> result;
    manifold.retract(x, α, p, result);

    CHECK(result[0] == x[0] + α);
    CHECK(result[7] == x[7]);

    Eigen::Matrix3<T> P = unpack3<T>(result.segment(1, 6));
    CHECK(Eigen::LLT<Eigen::Matrix3<T>>{P}.info() == Eigen::Success);
  }
}

TEMPLATE_TEST_CASE("Manifold - SPD derivatives match finite differences",
                   "[Manifold]", SCALAR_TYPES) {
  using T = TestType;
  using std::cbrt;
  using std::sqrt;

  constexpr T ε = std::numeric_limits<T>::epsilon();

  auto manifold = make_spd_manifold<T>();
  auto x = make_spd_point<T>();
  const int k = manifold.tangent_dimension(x.rows());

  auto R = [&](const Eigen::VectorX<T>& p) {
    Eigen::VectorX<T> result;
    manifold.retract(x, T(1), p, result);
    return result;
  };

  Eigen::MatrixX<T> J = manifold.jacobian(x);
  REQUIRE(J.rows() == x.rows());
  REQUIRE(J.cols() == k);

  const T h = cbrt(ε);
  for (int i = 0; i < k; ++i) {
    Eigen::VectorX<T> e = Eigen::VectorX<T>::Zero(k);
    e[i] = h;
    Eigen::VectorX<T> dR = (R(e) - R(-e)) / (T(2) * h);
    CHECK_THAT((dR - J.col(i)).template lpNorm<Eigen::Infinity>(),
               WithinAbs(T(0), T(10) * h * h));
  }

  Eigen::VectorX<T> v{
      {T(0.3), T(-1.2), T(0.4), T(2), T(-0.7), T(1.1), T(0.5), T(-0.8)}};

  // Only the lower triangle is stored
  Eigen::MatrixX<T> C_lower = manifold.hessian_correction(x, v);
  REQUIRE(C_lower.rows() == k);
  REQUIRE(C_lower.cols() == k);
  CHECK(C_lower.isLowerTriangular());
  Eigen::MatrixX<T> C =
      C_lower.template selfadjointView<Eigen::Lower>().toDenseMatrix();

  const T h2 = sqrt(sqrt(ε));
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      Eigen::VectorX<T> e_i = Eigen::VectorX<T>::Zero(k);
      Eigen::VectorX<T> e_j = Eigen::VectorX<T>::Zero(k);
      e_i[i] = h2;
      e_j[j] = h2;

      T d2 = (v.dot(R(e_i + e_j)) - v.dot(R(e_i - e_j)) - v.dot(R(-e_i + e_j)) +
              v.dot(R(-e_i - e_j))) /
             (T(4) * h2 * h2);
      CHECK_THAT(d2, WithinAbs(C(i, j), T(50) * sqrt(ε)));
    }
  }
}

TEMPLATE_TEST_CASE("Manifold - SPD barrier is linear along the retraction",
                   "[Manifold]", SCALAR_TYPES) {
  using T = TestType;
  using std::log;

  auto manifold = make_spd_manifold<T>();
  auto x = make_spd_point<T>();
  const int k = manifold.tangent_dimension(x.rows());

  CHECK(manifold.has_boundary());
  CHECK_FALSE(
      slp::Manifold<T>{{slp::ManifoldBlock<T>{slp::sphere_manifold<T>, 0, 3}}}
          .has_boundary());

  // b(x) = −log det P
  CHECK_THAT(manifold.barrier(x),
             WithinAbs(-log(unpack3<T>(x.segment(1, 6)).determinant()),
                       rounding_tolerance<T>(10)));

  // Euclidean coordinates don't contribute to the gradient
  Eigen::VectorX<T> g = manifold.barrier_gradient(x);
  REQUIRE(g.rows() == k);
  CHECK(g[0] == T(0));
  CHECK(g[7] == T(0));

  // b(R_x(αp)) = b(x) + α∇bᵀp exactly, including for large steps. Large steps
  // give ill-conditioned matrices, which amplify the rounding error of log det.
  Eigen::VectorX<T> p{
      {T(0.3), T(-1.2), T(0.4), T(2), T(-0.7), T(1.1), T(0.5), T(-0.8)}};
  for (T α : {T(0.01), T(1), T(5)}) {
    Eigen::VectorX<T> result;
    manifold.retract(x, α, p, result);
    CHECK_THAT(manifold.barrier(result),
               WithinAbs(manifold.barrier(x) + α * g.dot(p),
                         rounding_tolerance<T>(1e5)));
  }

  // A matrix that isn't positive definite has an infinite barrier
  Eigen::VectorX<T> indefinite{
      {T(0), T(1), T(0), T(-1), T(0), T(0), T(2), T(0)}};
  CHECK(manifold.barrier(indefinite) == std::numeric_limits<T>::infinity());
}
