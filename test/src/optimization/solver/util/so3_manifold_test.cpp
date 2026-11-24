// Copyright (c) Sleipnir contributors

#include <cmath>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/optimization/solver/util/manifold.hpp>
#include <sleipnir/optimization/solver/util/manifold_type.hpp>
#include <sleipnir/optimization/solver/util/so3_manifold.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Manifold - SO(3) retraction is a rotation", "[Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // R_q(p) = q ⊗ exp(p/2), so retracting the identity by θ about z gives the
  // rotation by θ about z
  slp::Manifold<T> manifold{
      {slp::ManifoldBlock<T>{slp::so3_manifold<T>, 0, 4}}};
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
