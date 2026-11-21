// Copyright (c) Sleipnir contributors

#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/optimization/solver/util/manifold.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

namespace {

// Euclidean, sphere S², Euclidean, SO(3), sphere S¹, SO(2)
template <typename T>
slp::Manifold<T> make_manifold() {
  return slp::Manifold<T>{{{slp::ManifoldType::SPHERE, 1, 3},
                           {slp::ManifoldType::SO3, 6, 4},
                           {slp::ManifoldType::SPHERE, 10, 2},
                           {slp::ManifoldType::SO2, 12, 2}}};
}

template <typename T>
Eigen::VectorX<T> make_point() {
  Eigen::VectorX<T> x{14};
  x << T(0.5), T(-0.3), T(0.8), T(0.2), T(1.5), T(-2), T(0.6), T(-0.2), T(0.7),
      T(0.1), T(-0.9), T(-0.4), T(-0.6), T(0.3);
  make_manifold<T>().project(x);
  return x;
}

}  // namespace

TEMPLATE_TEST_CASE("Manifold - Dimensions", "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  CHECK(slp::Manifold<T>{}.is_euclidean());
  CHECK(slp::Manifold<T>{}.tangent_dimension(5) == 5);

  auto manifold = make_manifold<T>();
  CHECK_FALSE(manifold.is_euclidean());

  // 14 − (3 − 2) − (4 − 3) − (2 − 1) − (2 − 1)
  CHECK(manifold.tangent_dimension(14) == 10);
}

TEMPLATE_TEST_CASE("Manifold - Projection", "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  auto manifold = make_manifold<T>();
  auto x = make_point<T>();

  CHECK(x[0] == T(0.5));
  CHECK_THAT(x.segment(1, 3).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK(x[4] == T(1.5));
  CHECK(x[5] == T(-2));
  CHECK_THAT(x.segment(6, 4).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK_THAT(x.segment(10, 2).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK_THAT(x.segment(12, 2).norm(), WithinAbs(T(1), T(1e-6)));

  // Zero blocks go to the default point
  Eigen::VectorX<T> zero = Eigen::VectorX<T>::Zero(14);
  manifold.project(zero);
  CHECK(zero[1] == T(1));
  CHECK(zero[6] == T(1));
  CHECK(zero[10] == T(1));
  CHECK(zero[12] == T(1));
  CHECK(zero.sum() == T(4));
}

TEMPLATE_TEST_CASE("Manifold - Retraction stays on manifold", "[Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto manifold = make_manifold<T>();
  auto x = make_point<T>();

  Eigen::VectorX<T> p{10};
  p << T(1), T(2), T(-3), T(0.5), T(-1), T(4), T(0.25), T(-2), T(3), T(0.7);

  Eigen::VectorX<T> result;
  manifold.retract(x, T(0.5), p, result);

  // Euclidean entries take the step
  CHECK(result[0] == x[0] + T(0.5) * p[0]);
  CHECK(result[4] == x[4] + T(0.5) * p[3]);
  CHECK(result[5] == x[5] + T(0.5) * p[4]);

  CHECK_THAT(result.segment(1, 3).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK_THAT(result.segment(6, 4).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK_THAT(result.segment(10, 2).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK_THAT(result.segment(12, 2).norm(), WithinAbs(T(1), T(1e-6)));

  // Zero step leaves x unchanged
  manifold.retract(x, T(0), p, result);
  CHECK_THAT((result - x).template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), T(1e-6)));
}

TEMPLATE_TEST_CASE("Manifold - SO(3) retraction is a rotation", "[Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // R_q(p) = q ⊗ exp(p/2), so retracting the identity by θ about z gives the
  // rotation by θ about z
  slp::Manifold<T> manifold{{{slp::ManifoldType::SO3, 0, 4}}};
  Eigen::VectorX<T> q{{T(1), T(0), T(0), T(0)}};
  Eigen::VectorX<T> p{{T(0), T(0), T(1)}};

  Eigen::VectorX<T> result;
  manifold.retract(q, T(0.5), p, result);

  using std::cos;
  using std::sin;
  CHECK_THAT(result[0], WithinAbs(cos(T(0.25)), T(1e-6)));
  CHECK_THAT(result[1], WithinAbs(T(0), T(1e-6)));
  CHECK_THAT(result[2], WithinAbs(T(0), T(1e-6)));
  CHECK_THAT(result[3], WithinAbs(sin(T(0.25)), T(1e-6)));
}

TEMPLATE_TEST_CASE("Manifold - Derivatives match finite differences",
                   "[Manifold]", SCALAR_TYPES) {
  using T = TestType;
  using std::cbrt;
  using std::sqrt;

  constexpr T ε = std::numeric_limits<T>::epsilon();

  auto manifold = make_manifold<T>();
  auto x = make_point<T>();
  const int k = manifold.tangent_dimension(x.rows());

  auto R = [&](const Eigen::VectorX<T>& p) {
    Eigen::VectorX<T> result;
    manifold.retract(x, T(1), p, result);
    return result;
  };

  Eigen::MatrixX<T> J = manifold.jacobian(x);
  REQUIRE(J.rows() == x.rows());
  REQUIRE(J.cols() == k);

  // Central differences of R along each tangent coordinate. Their error is
  // O(h²) truncation plus O(ε/h) rounding, which h = ∛ε balances at O(ε^(2/3)).
  const T h = cbrt(ε);
  for (int i = 0; i < k; ++i) {
    Eigen::VectorX<T> e = Eigen::VectorX<T>::Zero(k);
    e[i] = h;
    Eigen::VectorX<T> dR = (R(e) - R(-e)) / (T(2) * h);
    CHECK_THAT((dR - J.col(i)).template lpNorm<Eigen::Infinity>(),
               WithinAbs(T(0), T(10) * h * h));
  }

  // The second directional derivative of vᵀR along p is pᵀCp, where C is the
  // Hessian correction for v
  Eigen::VectorX<T> v{x.rows()};
  v << T(0.3), T(-1.2), T(0.4), T(2), T(-0.7), T(1.1), T(0.5), T(-0.8), T(1.3),
      T(0.9), T(-0.6), T(1.7), T(-1.4), T(0.8);
  Eigen::MatrixX<T> C = manifold.hessian_correction(x, v);
  REQUIRE(C.rows() == k);
  REQUIRE(C.cols() == k);

  // Mixed second differences have O(h²) truncation plus O(ε/h²) rounding
  // error, which h = ∜ε balances at O(√ε)
  const T h2 = sqrt(sqrt(ε));
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      Eigen::VectorX<T> e_i = Eigen::VectorX<T>::Zero(k);
      Eigen::VectorX<T> e_j = Eigen::VectorX<T>::Zero(k);
      e_i[i] = h2;
      e_j[j] = h2;

      // Mixed second derivative of vᵀR(p) at p = 0. Euclidean coordinates
      // contribute nothing since R is linear in them.
      T d2 = (v.dot(R(e_i + e_j)) - v.dot(R(e_i - e_j)) - v.dot(R(-e_i + e_j)) +
              v.dot(R(-e_i - e_j))) /
             (T(4) * h2 * h2);
      CHECK_THAT(d2, WithinAbs(C(i, j), T(50) * sqrt(ε)));
    }
  }

  // relative_step_inf_norm() is ‖Jp/(1 + |x|)‖_∞
  Eigen::VectorX<T> p = Eigen::VectorX<T>::LinSpaced(k, T(-1), T(2));
  T expected = ((J * p).array().abs() / (T(1) + x.array().abs())).maxCoeff();
  CHECK_THAT(manifold.relative_step_inf_norm(x, p),
             WithinAbs(expected, rounding_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Manifold - SO(2) retraction is a rotation", "[Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // R_z(p) = z exp(ip), so retracting the rotation by 0.3 rad by 0.5 rad gives
  // the rotation by 0.8 rad
  using std::cos;
  using std::sin;
  slp::Manifold<T> manifold{{{slp::ManifoldType::SO2, 0, 2}}};
  Eigen::VectorX<T> z{{cos(T(0.3)), sin(T(0.3))}};
  Eigen::VectorX<T> p{{T(1)}};

  Eigen::VectorX<T> result;
  manifold.retract(z, T(0.5), p, result);

  CHECK_THAT(result[0], WithinAbs(cos(T(0.8)), T(1e-6)));
  CHECK_THAT(result[1], WithinAbs(sin(T(0.8)), T(1e-6)));
}

TEMPLATE_TEST_CASE("Manifold - Sphere tangent basis is orthonormal",
                   "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  // Includes points with negative and zero first coordinates
  auto x0 = GENERATE(Eigen::Vector3d{1, 0, 0}, Eigen::Vector3d{-1, 0, 0},
                     Eigen::Vector3d{0, 1, 0}, Eigen::Vector3d{0.3, -0.5, 0.8},
                     Eigen::Vector3d{-0.3, -0.5, -0.8});

  slp::Manifold<T> manifold{{{slp::ManifoldType::SPHERE, 0, 3}}};
  Eigen::VectorX<T> x = x0.template cast<T>();
  manifold.project(x);

  Eigen::MatrixX<T> B = manifold.jacobian(x);
  CHECK_THAT((B.transpose() * B - Eigen::MatrixX<T>::Identity(2, 2))
                 .template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), T(1e-6)));
  CHECK_THAT((B.transpose() * x).template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), T(1e-6)));
}

namespace {

// Euclidean, 3×3 SPD, Euclidean
template <typename T>
slp::Manifold<T> make_spd_manifold() {
  return slp::Manifold<T>{{{slp::ManifoldType::SPD, 1, 6}}};
}

template <typename T>
Eigen::VectorX<T> make_spd_point() {
  // Lower triangle in row-major order of
  //
  //   [ 2   −0.5  0.3]
  //   [−0.5  1.5  0.2]
  //   [ 0.3  0.2  0.8]
  Eigen::VectorX<T> x{8};
  x << T(0.7), T(2), T(-0.5), T(1.5), T(0.3), T(0.2), T(0.8), T(-1.2);
  return x;
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
  Eigen::VectorX<T> indefinite{8};
  indefinite << T(0), T(1), T(0), T(-1), T(0), T(0), T(2), T(0);
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
    Eigen::VectorX<T> x{8};
    x << T(0), P(0, 0), P(1, 0), P(1, 1), P(2, 0), P(2, 1), P(2, 2), T(0);
    return x;
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

  Eigen::VectorX<T> v{x.rows()};
  v << T(0.3), T(-1.2), T(0.4), T(2), T(-0.7), T(1.1), T(0.5), T(-0.8);

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
  CHECK_FALSE(make_manifold<T>().has_boundary());

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
  Eigen::VectorX<T> p{k};
  p << T(0.3), T(-1.2), T(0.4), T(2), T(-0.7), T(1.1), T(0.5), T(-0.8);
  for (T α : {T(0.01), T(1), T(5)}) {
    Eigen::VectorX<T> result;
    manifold.retract(x, α, p, result);
    CHECK_THAT(manifold.barrier(result),
               WithinAbs(manifold.barrier(x) + α * g.dot(p),
                         rounding_tolerance<T>(1e5)));
  }

  // A matrix that isn't positive definite has an infinite barrier
  Eigen::VectorX<T> indefinite{8};
  indefinite << T(0), T(1), T(0), T(-1), T(0), T(0), T(2), T(0);
  CHECK(manifold.barrier(indefinite) == std::numeric_limits<T>::infinity());
}
