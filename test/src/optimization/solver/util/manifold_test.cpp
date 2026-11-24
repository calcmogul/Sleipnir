// Copyright (c) Sleipnir contributors

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/optimization/solver/util/manifold.hpp>
#include <sleipnir/optimization/solver/util/manifold_type.hpp>
#include <sleipnir/optimization/solver/util/sphere_manifold.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

namespace {

// Euclidean, sphere S², Euclidean, sphere S¹, Euclidean
template <typename T>
slp::Manifold<T> make_manifold() {
  return slp::Manifold<T>{
      {slp::ManifoldBlock<T>{slp::sphere_manifold<T>, 1, 3},
       slp::ManifoldBlock<T>{slp::sphere_manifold<T>, 10, 2}}};
}

template <typename T>
Eigen::VectorX<T> make_point() {
  Eigen::VectorX<T> x{{T(0.5), T(-0.3), T(0.8), T(0.2), T(1.5), T(-2), T(0.6),
                       T(-0.2), T(0.7), T(0.1), T(-0.9), T(-0.4), T(-0.6),
                       T(0.3)}};
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

  // 14 − (3 − 2) − (2 − 1)
  CHECK(manifold.tangent_dimension(14) == 12);
}

TEMPLATE_TEST_CASE("Manifold - Projection", "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  auto manifold = make_manifold<T>();
  auto x = make_point<T>();

  CHECK(x[0] == T(0.5));
  CHECK_THAT(x.segment(1, 3).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK(x[4] == T(1.5));
  CHECK(x[5] == T(-2));
  CHECK(x[6] == T(0.6));
  CHECK_THAT(x.segment(10, 2).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK(x[12] == T(-0.6));

  // Zero blocks go to the default point
  Eigen::VectorX<T> zero = Eigen::VectorX<T>::Zero(14);
  manifold.project(zero);
  CHECK(zero[1] == T(1));
  CHECK(zero[10] == T(1));
  CHECK(zero.sum() == T(2));
}

TEMPLATE_TEST_CASE("Manifold - Retraction stays on manifold", "[Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto manifold = make_manifold<T>();
  auto x = make_point<T>();

  Eigen::VectorX<T> p{{T(1), T(2), T(-3), T(0.5), T(-1), T(4), T(0.25), T(-2),
                       T(3), T(0.7), T(-0.4), T(1.2)}};

  Eigen::VectorX<T> result;
  manifold.retract(x, T(0.5), p, result);

  // Euclidean entries take the step
  CHECK(result[0] == x[0] + T(0.5) * p[0]);
  CHECK(result[4] == x[4] + T(0.5) * p[3]);
  CHECK(result[5] == x[5] + T(0.5) * p[4]);
  CHECK(result[9] == x[9] + T(0.5) * p[8]);
  CHECK(result[12] == x[12] + T(0.5) * p[10]);

  CHECK_THAT(result.segment(1, 3).norm(), WithinAbs(T(1), T(1e-6)));
  CHECK_THAT(result.segment(10, 2).norm(), WithinAbs(T(1), T(1e-6)));

  // Zero step leaves x unchanged
  manifold.retract(x, T(0), p, result);
  CHECK_THAT((result - x).template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), T(1e-6)));
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
  Eigen::VectorX<T> v{{T(0.3), T(-1.2), T(0.4), T(2), T(-0.7), T(1.1), T(0.5),
                       T(-0.8), T(1.3), T(0.9), T(-0.6), T(1.7), T(-1.4),
                       T(0.8)}};
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
