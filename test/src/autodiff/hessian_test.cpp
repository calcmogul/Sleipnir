// Copyright (c) Sleipnir contributors

#include <cmath>
#include <functional>
#include <numeric>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "range.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Hessian - Linear", "[Hessian]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x
  slp::VariableMatrix<T> x{1};
  x[0].set_value(T(3));
  slp::Variable y = x[0];

  // dy/dx = 1
  T g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == T(1));

  // d²y/dx² = 0
  auto H = slp::Hessian(y, x);
  CHECK(slp::value(H.get()).coeff(0, 0) == T(0));
  CHECK(H.value().coeff(0, 0) == T(0));
}

TEMPLATE_TEST_CASE("Hessian - Quadratic", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x²
  slp::VariableMatrix<T> x{1};
  x[0].set_value(T(3));
  slp::Variable y = x[0] * x[0];

  // dy/dx = 2x = 6
  T g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == T(6));

  // d²y/dx² = 2
  auto H = slp::Hessian(y, x);
  CHECK(slp::value(H.get()).coeff(0, 0) == T(2));
  CHECK(H.value().coeff(0, 0) == T(2));
}

TEMPLATE_TEST_CASE("Hessian - Cubic", "[Hessian]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x³
  slp::VariableMatrix<T> x{1};
  x[0].set_value(T(3));
  slp::Variable y = x[0] * x[0] * x[0];

  // dy/dx = 3x² = 27
  T g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == T(27));

  // d²y/dx² = 6x = 18
  auto H = slp::Hessian(y, x);
  CHECK(slp::value(H.get()).coeff(0, 0) == T(18));
  CHECK(H.value().coeff(0, 0) == T(18));
}

TEMPLATE_TEST_CASE("Hessian - Quartic", "[Hessian]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x⁴
  slp::VariableMatrix<T> x{1};
  x[0].set_value(T(3));
  slp::Variable y = x[0] * x[0] * x[0] * x[0];

  // dy/dx = 4x³ = 108
  T g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == T(108));

  // d²y/dx² = 12x² = 108
  auto H = slp::Hessian(y, x);
  CHECK(slp::value(H.get()).coeff(0, 0) == T(108));
  CHECK(H.value().coeff(0, 0) == T(108));
}

TEMPLATE_TEST_CASE("Hessian - Sum", "[Hessian]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{5};
  x[0].set_value(T(1));
  x[1].set_value(T(2));
  x[2].set_value(T(3));
  x[3].set_value(T(4));
  x[4].set_value(T(5));

  // y = sum(x)
  auto y = std::accumulate(x.begin(), x.end(), slp::Variable{T(0)});
  CHECK(y.value() == T(15));

  auto g = slp::Gradient(y, x);
  CHECK(slp::value(g.get()).toDense() ==
        Eigen::Matrix<T, 5, 1>::Constant(T(1)));
  CHECK(g.value().toDense() == Eigen::Matrix<T, 5, 1>::Constant(T(1)));

  auto H = slp::Hessian(y, x);
  CHECK(slp::value(H.get()).toDense() == Eigen::Matrix<T, 5, 5>::Zero());
  CHECK(H.value().toDense() == Eigen::Matrix<T, 5, 5>::Zero());
}

TEMPLATE_TEST_CASE("Hessian - Sum of products", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{5};
  for (int i = 0; i < 5; ++i) {
    x[i].set_value(T(i + 1));
  }

  // y = ||x||²
  slp::Variable y = x.T() * x;
  CHECK(y.value() == T(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5));

  auto g = slp::Gradient(y, x);
  CHECK(slp::value(g.get()).toDense() == T(2) * x.value());
  CHECK(g.value().toDense() == T(2) * x.value());

  auto H = slp::Hessian(y, x);

  Eigen::Matrix<T, 5, 5> expected_H =
      Eigen::Vector<T, 5>::Constant(T(2)).asDiagonal();
  CHECK(slp::value(H.get()).toDense() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Product of sines", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::sin;
  using std::tan;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{5};
  for (int i = 0; i < 5; ++i) {
    x[i].set_value(T(i + 1));
  }

  // y = prod(sin(x))
  auto temp = x.cwise_transform(slp::sin<T>);
  auto y = std::accumulate(temp.begin(), temp.end(), slp::Variable{T(1)},
                           std::multiplies{});
  CHECK_THAT(y.value(), WithinAbs(sin(T(1)) * sin(T(2)) * sin(T(3)) *
                                      sin(T(4)) * sin(T(5)),
                                  T(1e-15)));

  auto g = slp::Gradient(y, x);
  for (int i = 0; i < x.rows(); ++i) {
    CHECK_THAT(slp::value(g.get()).coeff(i),
               WithinAbs(y.value() / tan(x[i].value()), T(1e-15)));
    CHECK_THAT(g.value().coeff(i),
               WithinAbs(y.value() / tan(x[i].value()), T(1e-15)));
  }

  auto H = slp::Hessian(y, x);

  Eigen::Matrix<T, 5, 5> expected_H;
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      if (i == j) {
        expected_H(i, j) = -y.value();
      } else {
        expected_H(i, j) = y.value() / (tan(x[i].value()) * tan(x[j].value()));
      }
    }
  }
  CHECK_THAT(slp::value(H.get()), MatrixWithinAbs(expected_H, T(1e-15)));
  CHECK_THAT(H.value().toDense(), MatrixWithinAbs(expected_H, T(1e-15)));
}

TEMPLATE_TEST_CASE("Hessian - Sum of squared residuals", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{5};
  for (int i = 0; i < 5; ++i) {
    x[i].set_value(T(1));
  }

  // y = sum(diff(x).^2)
  auto temp = (x.block(0, 0, 4, 1) - x.block(1, 0, 4, 1))
                  .cwise_transform([](const slp::Variable<T>& x) {
                    return slp::pow(x, T(2));
                  });
  auto y = std::accumulate(temp.begin(), temp.end(), slp::Variable{T(0)});
  auto g = slp::Gradient(y, x).value().toDense();

  CHECK(y.value() == T(0));
  CHECK(g[0] == T(2) * x[0].value() - T(2) * x[1].value());
  CHECK(g[1] ==
        T(-2) * x[0].value() + T(4) * x[1].value() - T(2) * x[2].value());
  CHECK(g[2] ==
        T(-2) * x[1].value() + T(4) * x[2].value() - T(2) * x[3].value());
  CHECK(g[3] ==
        T(-2) * x[2].value() + T(4) * x[3].value() - T(2) * x[4].value());
  CHECK(g[4] == T(-2) * x[3].value() + T(2) * x[4].value());

  auto H = slp::Hessian(y, x);

  Eigen::Matrix<T, 5, 5> expected_H{{T(2), T(-2), T(0), T(0), T(0)},
                                    {T(-2), T(4), T(-2), T(0), T(0)},
                                    {T(0), T(-2), T(4), T(-2), T(0)},
                                    {T(0), T(0), T(-2), T(4), T(-2)},
                                    {T(0), T(0), T(0), T(-2), T(2)}};
  CHECK(slp::value(H.get()).toDense() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Sum of squares", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> r{4};
  r.set_value(Eigen::Vector<T, 4>{{T(25), T(10), T(5), T(0)}});

  slp::VariableMatrix<T> x{4};
  for (int i = 0; i < 4; ++i) {
    x[i].set_value(T(0));
  }

  slp::Variable J = T(0);
  for (int i = 0; i < 4; ++i) {
    J += (r[i] - x[i]) * (r[i] - x[i]);
  }

  auto H = slp::Hessian(J, x);

  Eigen::Matrix<T, 4, 4> expected_H =
      Eigen::Vector<T, 4>::Constant(T(2)).asDiagonal();
  CHECK(slp::value(H.get()).toDense() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Nested powers", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr T x0(3);

  slp::Variable<T> x;
  x.set_value(x0);

  auto y = slp::pow(slp::pow(x, T(2)), T(2));

  auto J = slp::Jacobian(y, x).value().toDense();
  CHECK_THAT(J(0, 0), WithinAbs(T(4) * x0 * x0 * x0, T(1e-12)));

  auto H = slp::Hessian(y, x).value().toDense();
  CHECK_THAT(H(0, 0), WithinAbs(T(12) * x0 * x0, T(1e-12)));
}

TEMPLATE_TEST_CASE("Hessian - Rosenbrock", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  // z = (1 − x)² + 100(y − x²)²
  //   = 100(−x² + y)² + (−x + 1)²
  //
  // ∂z/∂x = 200(−x² + y)⋅−2x + 2(−x + 1)⋅−1
  //       = −400x(−x² + y) − 2(−x + 1)
  //       = 400x³ − 400xy + 2x − 2
  //
  // ∂z/∂y = 200(−x² + y)
  //
  // ∂²z/∂x² = 1200x² − 400y + 2
  // ∂²z/∂xy = −400x
  // ∂²z/∂y² = 200

  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> input{2};
  auto& x = input[0];
  auto& y = input[1];
  auto hessian = slp::Hessian(
      pow(T(1) - x, T(2)) + T(100) * pow(y - pow(x, T(2)), T(2)), input);

  for (auto x0 : range(T(-2.5), T(2.5), T(0.1))) {
    for (auto y0 : range(T(-2.5), T(2.5), T(0.1))) {
      x.set_value(x0);
      y.set_value(y0);

      auto H = hessian.value().toDense();
      CHECK_THAT(H(0, 0),
                 WithinAbs(T(1200) * x0 * x0 - T(400) * y0 + T(2), T(1e-11)));
      CHECK(H(0, 1) == T(-400) * x0);
      CHECK(H(1, 0) == T(-400) * x0);
      CHECK(H(1, 1) == T(200));
    }
  }
}

TEMPLATE_TEST_CASE("Hessian - Edge pushing (Wang) example 1", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  // See example 1 of [1]
  //
  // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
  //     efficient Hessian computation via automatic differentiation", 2016.
  //     https://sci-hub.st/10.1007/s12532-016-0100-3

  using T = TestType;
  using std::cos;
  using std::sin;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{2};
  x[0].set_value(T(3));
  x[1].set_value(T(4));

  // y = (x₀sin(x₁)) x₀
  auto y = (x[0] * slp::sin(x[1])) * x[0];

  // dy/dx = [2x₀sin(x₁)  x₀²cos(x₁)]
  // dy/dx = [ 6sin(4)     9cos(4)  ]
  auto J = slp::Jacobian(y, x);
  Eigen::Matrix<T, 1, 2> expected_J{{T(6) * sin(T(4)), T(9) * cos(T(4))}};
  CHECK(slp::value(J.get()).toDense() == expected_J);
  CHECK(J.value().toDense() == expected_J);

  //           [ 2sin(x₁)    2x₀cos(x₁)]
  // d²y/dx² = [2x₀cos(x₁)  −x₀²sin(x₁)]
  //
  //           [2sin(4)   6cos(4)]
  // d²y/dx² = [6cos(4)  −9sin(4)]
  auto H = slp::Hessian(y, x);
  Eigen::Matrix<T, 2, 2> expected_H{{T(2) * sin(T(4)), T(6) * cos(T(4))},
                                    {T(6) * cos(T(4)), T(-9) * sin(T(4))}};
  CHECK(slp::value(H.get()).toDense() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Edge pushing (Petro) figure 1", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  // See figure 1 of [1]
  //
  // [1] Petro, C. G., et al. "On efficient Hessian computation using the edge
  //     pushing algorithm in Julia", 2017.
  //     https://mlubin.github.io/pdf/edge_pushing_julia.pdf

  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = p₁ log(x₁x₂)
  slp::Variable<T> p_1 = 2.0;
  slp::VariableMatrix<T> x{2};
  x[0].set_value(T(2));
  x[1].set_value(T(3));
  slp::Variable<T> y = p_1 * slp::log(x[0] * x[1]);

  // d²y/dx² = [−p₁/x₁²     0   ]
  //           [   0     −p₁/x₂²]
  slp::Hessian H{y, x};
  Eigen::Matrix<T, 2, 2> expected_H{
      {-p_1.value() / (x[0].value() * x[0].value()), T(0)},
      {T(0), -p_1.value() / (x[1].value() * x[1].value())}};
  CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Variable reuse", "[Hessian]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> y;
  slp::VariableMatrix<T> x{1};

  // y = x³
  x[0].set_value(T(1));
  y = x[0] * x[0] * x[0];

  slp::Hessian hessian{y, x};

  // d²y/dx² = 6x
  // H = 6
  auto H = hessian.value().toDense();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == T(6));

  x[0].set_value(T(2));
  // d²y/dx² = 6x
  // H = 12
  H = hessian.value();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == T(12));
}
