// Copyright (c) Sleipnir contributors

#include <cmath>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/optimization/solver/util/manifold.hpp>
#include <sleipnir/optimization/solver/util/manifold_type.hpp>
#include <sleipnir/optimization/solver/util/so2_manifold.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Manifold - SO(2) retraction is a rotation", "[Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // R_z(p) = z exp(ip), so retracting the rotation by 0.3 rad by 0.5 rad gives
  // the rotation by 0.8 rad
  using std::cos;
  using std::sin;
  slp::Manifold<T> manifold{
      {slp::ManifoldBlock<T>{slp::so2_manifold<T>, 0, 2}}};
  Eigen::VectorX<T> z{{cos(T(0.3)), sin(T(0.3))}};
  Eigen::VectorX<T> p{{T(1)}};

  Eigen::VectorX<T> result;
  manifold.retract(z, T(0.5), p, result);

  CHECK_THAT(result[0], WithinAbs(cos(T(0.8)), T(1e-6)));
  CHECK_THAT(result[1], WithinAbs(sin(T(0.8)), T(1e-6)));
}
