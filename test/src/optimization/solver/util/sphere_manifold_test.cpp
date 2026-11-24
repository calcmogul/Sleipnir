// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/optimization/solver/util/manifold.hpp>
#include <sleipnir/optimization/solver/util/manifold_type.hpp>
#include <sleipnir/optimization/solver/util/sphere_manifold.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Manifold - Sphere tangent basis is orthonormal",
                   "[Manifold]", SCALAR_TYPES) {
  using T = TestType;

  // Includes points with negative and zero first coordinates
  auto x0 = GENERATE(Eigen::Vector3d{1, 0, 0}, Eigen::Vector3d{-1, 0, 0},
                     Eigen::Vector3d{0, 1, 0}, Eigen::Vector3d{0.3, -0.5, 0.8},
                     Eigen::Vector3d{-0.3, -0.5, -0.8});

  slp::Manifold<T> manifold{
      {slp::ManifoldBlock<T>{slp::sphere_manifold<T>, 0, 3}}};
  Eigen::VectorX<T> x = x0.template cast<T>();
  manifold.project(x);

  Eigen::MatrixX<T> B = manifold.jacobian(x);
  CHECK_THAT((B.transpose() * B - Eigen::MatrixX<T>::Identity(2, 2))
                 .template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), T(1e-6)));
  CHECK_THAT((B.transpose() * x).template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), T(1e-6)));
}
