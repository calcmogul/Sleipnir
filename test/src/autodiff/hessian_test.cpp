// Copyright (c) Sleipnir contributors

#include <cmath>
#include <functional>
#include <numbers>
#include <numeric>
#include <utility>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/util/pool.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "range.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Hessian - Linear", "[Hessian]", SCALAR_TYPES) {
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
  CHECK(H.get().value(0, 0) == T(0));
  CHECK(H.value().coeff(0, 0) == T(0));
}

TEMPLATE_TEST_CASE("Hessian - Quadratic", "[Hessian]", SCALAR_TYPES) {
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
  CHECK(H.get().value(0, 0) == T(2));
  CHECK(H.value().coeff(0, 0) == T(2));
}

TEMPLATE_TEST_CASE("Hessian - Cubic", "[Hessian]", SCALAR_TYPES) {
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
  CHECK(H.get().value(0, 0) == T(18));
  CHECK(H.value().coeff(0, 0) == T(18));
}

TEMPLATE_TEST_CASE("Hessian - Quartic", "[Hessian]", SCALAR_TYPES) {
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
  CHECK(H.get().value(0, 0) == T(108));
  CHECK(H.value().coeff(0, 0) == T(108));
}

TEMPLATE_TEST_CASE("Hessian - Sum", "[Hessian]", SCALAR_TYPES) {
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
  CHECK(g.get().value() == Eigen::Matrix<T, 5, 1>::Constant(T(1)));
  CHECK(g.value().toDense() == Eigen::Matrix<T, 5, 1>::Constant(T(1)));

  auto H = slp::Hessian(y, x);
  CHECK(H.get().value() == Eigen::Matrix<T, 5, 5>::Zero());
  CHECK(H.value().toDense() == Eigen::Matrix<T, 5, 5>::Zero());
}

TEMPLATE_TEST_CASE("Hessian - Sum of products", "[Hessian]", SCALAR_TYPES) {
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
  CHECK(g.get().value() == T(2) * x.value());
  CHECK(g.value().toDense() == T(2) * x.value());

  auto H = slp::Hessian(y, x);

  Eigen::Matrix<T, 5, 5> expected_H =
      Eigen::Vector<T, 5>::Constant(T(2)).asDiagonal();
  CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Product of sines", "[Hessian]", SCALAR_TYPES) {
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
    CHECK_THAT(g.get().value(i),
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
  CHECK_THAT(H.get().value(), WithinAbs(expected_H, T(1e-15)));
  CHECK_THAT(H.value().toDense(), WithinAbs(expected_H, T(1e-15)));
}

TEMPLATE_TEST_CASE("Hessian - Sum of squared residuals", "[Hessian]",
                   SCALAR_TYPES) {
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
  CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Sum of squares", "[Hessian]", SCALAR_TYPES) {
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
  CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Nested powers", "[Hessian]", SCALAR_TYPES) {
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

// Checks the Hessian of y = f(u) where u = x₀x₁
//
// ∂²y/∂x₀² = f″(u) x₁²
// ∂²y/∂x₀∂x₁ = f″(u) x₀x₁ + f′(u)
// ∂²y/∂x₁² = f″(u) x₀²
template <typename T>
void check_unary_hessian(
    std::function<slp::Variable<T>(const slp::Variable<T>&)> f,
    std::function<T(T)> df, std::function<T(T)> d2f) {
  slp::VariableMatrix<T> x{2};
  x[0].set_value(T(0.6));
  x[1].set_value(T(0.7));
  slp::Hessian hessian{f(x[0] * x[1]), x};
  auto symbolic_H = hessian.get();

  // Check symbolic Hessian at a point other than where it was built too
  for (auto [x0, x1] : {std::pair{T(0.6), T(0.7)}, std::pair{T(0.3), T(0.7)}}) {
    x[0].set_value(x0);
    x[1].set_value(x1);

    T u = x0 * x1;
    T mixed = d2f(u) * u + df(u);
    Eigen::Matrix<T, 2, 2> expected{{d2f(u) * x1 * x1, mixed},
                                    {mixed, d2f(u) * x0 * x0}};
    CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
    CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));
  }
}

TEMPLATE_TEST_CASE("Hessian - Division", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x₀/x₁
  //
  // ∂²y/∂x₀² = 0
  // ∂²y/∂x₀∂x₁ = −1/x₁²
  // ∂²y/∂x₁² = 2x₀/x₁³
  slp::VariableMatrix<T> x{2};
  slp::Hessian hessian{x[0] / x[1], x};
  auto symbolic_H = hessian.get();

  for (auto [x0, x1] : {std::pair{T(3), T(2)}, std::pair{T(-1.5), T(0.5)}}) {
    x[0].set_value(x0);
    x[1].set_value(x1);

    T mixed = T(-1) / (x1 * x1);
    Eigen::Matrix<T, 2, 2> expected{{T(0), mixed},
                                    {mixed, T(2) * x0 / (x1 * x1 * x1)}};
    CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
    CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));
  }
}

TEMPLATE_TEST_CASE("Hessian - Unary minus", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return -u; },
                         [](T) { return T(-1); }, [](T) { return T(0); });
}

TEMPLATE_TEST_CASE("Hessian - abs()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::abs(u); },
                         [](T) { return T(1); }, [](T) { return T(0); });
}

TEMPLATE_TEST_CASE("Hessian - acos()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::sqrt;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::acos(u); },
                         [](T u) { return T(-1) / sqrt(T(1) - u * u); },
                         [](T u) {
                           T s = sqrt(T(1) - u * u);
                           return -u / (s * s * s);
                         });
}

TEMPLATE_TEST_CASE("Hessian - asin()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::sqrt;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::asin(u); },
                         [](T u) { return T(1) / sqrt(T(1) - u * u); },
                         [](T u) {
                           T s = sqrt(T(1) - u * u);
                           return u / (s * s * s);
                         });
}

TEMPLATE_TEST_CASE("Hessian - atan()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>(
      [](const auto& u) { return slp::atan(u); },
      [](T u) { return T(1) / (T(1) + u * u); },
      [](T u) { return T(-2) * u / ((T(1) + u * u) * (T(1) + u * u)); });
}

TEMPLATE_TEST_CASE("Hessian - atan2()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // z = atan2(y, x)
  //
  // ∂²z/∂y² = −2xy/(x² + y²)²
  // ∂²z/∂y∂x = (y² − x²)/(x² + y²)²
  // ∂²z/∂x² = 2xy/(x² + y²)²
  slp::VariableMatrix<T> input{2};
  slp::Hessian hessian{slp::atan2(input[0], input[1]), input};
  auto symbolic_H = hessian.get();

  for (auto [y, x] : {std::pair{T(1.5), T(0.5)}, std::pair{T(-0.7), T(1.3)}}) {
    input[0].set_value(y);
    input[1].set_value(x);

    T d = x * x + y * y;
    T mixed = (y * y - x * x) / (d * d);
    Eigen::Matrix<T, 2, 2> expected{{T(-2) * x * y / (d * d), mixed},
                                    {mixed, T(2) * x * y / (d * d)}};
    CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
    CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));
  }
}

TEMPLATE_TEST_CASE("Hessian - cbrt()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cbrt;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>(
      [](const auto& u) { return slp::cbrt(u); },
      [](T u) { return T(1) / (T(3) * cbrt(u) * cbrt(u)); },
      [](T u) { return T(-2) / (T(9) * u * cbrt(u) * cbrt(u)); });
}

TEMPLATE_TEST_CASE("Hessian - cos()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cos;
  using std::sin;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::cos(u); },
                         [](T u) { return -sin(u); },
                         [](T u) { return -cos(u); });
}

TEMPLATE_TEST_CASE("Hessian - cosh()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cosh;
  using std::sinh;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::cosh(u); },
                         [](T u) { return sinh(u); },
                         [](T u) { return cosh(u); });
}

TEMPLATE_TEST_CASE("Hessian - erf()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::exp;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>(
      [](const auto& u) { return slp::erf(u); },
      [](T u) { return T(2.0 * std::numbers::inv_sqrtpi) * exp(-u * u); },
      [](T u) { return T(-4.0 * std::numbers::inv_sqrtpi) * u * exp(-u * u); });
}

TEMPLATE_TEST_CASE("Hessian - exp()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::exp;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::exp(u); },
                         [](T u) { return exp(u); },
                         [](T u) { return exp(u); });
}

TEMPLATE_TEST_CASE("Hessian - hypot()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::sqrt;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = ‖v‖ where v = [x₀, x₁, x₂]ᵀ
  //
  // ∂²y/∂xᵢ∂xⱼ = (‖v‖²δᵢⱼ − xᵢxⱼ)/‖v‖³
  auto expected_H = [](const Eigen::Vector<T, 3>& v) {
    T r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    T r3 = r2 * sqrt(r2);

    Eigen::Matrix<T, 3, 3> H;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        H(i, j) = ((i == j ? r2 : T(0)) - v[i] * v[j]) / r3;
      }
    }
    return H;
  };

  SECTION("Two arguments") {
    slp::VariableMatrix<T> x{2};
    slp::Hessian hessian{slp::hypot(x[0], x[1]), x};
    auto symbolic_H = hessian.get();

    for (auto [x0, x1] :
         {std::pair{T(1.3), T(2.3)}, std::pair{T(-0.5), T(1.5)}}) {
      x[0].set_value(x0);
      x[1].set_value(x1);

      Eigen::Matrix<T, 2, 2> expected =
          expected_H(Eigen::Vector<T, 3>{x0, x1, T(0)})
              .template topLeftCorner<2, 2>();
      CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
      CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));
    }
  }

  SECTION("Three arguments") {
    slp::VariableMatrix<T> x{3};
    slp::Hessian hessian{slp::hypot(x[0], x[1], x[2]), x};
    auto symbolic_H = hessian.get();

    for (auto v : {Eigen::Vector<T, 3>{T(1.3), T(2.3), T(3.3)},
                   Eigen::Vector<T, 3>{T(-0.5), T(1.5), T(-2.5)}}) {
      x[0].set_value(v[0]);
      x[1].set_value(v[1]);
      x[2].set_value(v[2]);

      auto expected = expected_H(v);
      CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
      CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));
    }
  }

  SECTION("Repeated argument") {
    // y = hypot(a, b, a) = √(2a² + b²)
    //
    // ∂²y/∂a² = (2y² − 4a²)/y³
    // ∂²y/∂a∂b = −2ab/y³
    // ∂²y/∂b² = (y² − b²)/y³
    slp::VariableMatrix<T> x{2};
    slp::Hessian hessian{slp::hypot(x[0], x[1], x[0]), x};
    auto symbolic_H = hessian.get();

    for (auto [a, b] :
         {std::pair{T(1.3), T(2.3)}, std::pair{T(-0.5), T(1.5)}}) {
      x[0].set_value(a);
      x[1].set_value(b);

      T y2 = T(2) * a * a + b * b;
      T y3 = y2 * sqrt(y2);
      T mixed = T(-2) * a * b / y3;
      Eigen::Matrix<T, 2, 2> expected{{(T(2) * y2 - T(4) * a * a) / y3, mixed},
                                      {mixed, (y2 - b * b) / y3}};
      CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
      CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));
    }
  }
}

TEMPLATE_TEST_CASE("Hessian - if_else()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x₀²x₁ if x₀ < x₁, otherwise x₁³
  //
  // The condition's arguments are decision variables, but they don't
  // contribute to the Hessian.
  slp::VariableMatrix<T> x{2};
  auto f = slp::if_else([](T a, T b) { return a < b; }, x[0], x[1],
                        x[0] * x[0] * x[1], x[1] * x[1] * x[1]);
  slp::Hessian hessian{f, x};
  auto symbolic_H = hessian.get();

  // True branch
  //
  // ∂²y/∂x₀² = 2x₁
  // ∂²y/∂x₀∂x₁ = 2x₀
  // ∂²y/∂x₁² = 0
  x[0].set_value(T(2));
  x[1].set_value(T(3));
  Eigen::Matrix<T, 2, 2> expected{{T(6), T(4)}, {T(4), T(0)}};
  CHECK(symbolic_H.value() == expected);
  CHECK(hessian.value().toDense() == expected);

  // False branch
  //
  // ∂²y/∂x₀² = 0
  // ∂²y/∂x₀∂x₁ = 0
  // ∂²y/∂x₁² = 6x₁
  x[0].set_value(T(3));
  x[1].set_value(T(2));
  expected = Eigen::Matrix<T, 2, 2>{{T(0), T(0)}, {T(0), T(12)}};
  CHECK(symbolic_H.value() == expected);
  CHECK(hessian.value().toDense() == expected);
}

TEMPLATE_TEST_CASE("Hessian - log()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::log(u); },
                         [](T u) { return T(1) / u; },
                         [](T u) { return T(-1) / (u * u); });
}

TEMPLATE_TEST_CASE("Hessian - log10()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>(
      [](const auto& u) { return slp::log10(u); },
      [](T u) { return T(1) / (T(std::numbers::ln10) * u); },
      [](T u) { return T(-1) / (T(std::numbers::ln10) * u * u); });
}

TEMPLATE_TEST_CASE("Hessian - max() and min()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> input{2};
  Eigen::Matrix<T, 2, 2> expected_y{{T(0), T(0)}, {T(0), T(2)}};
  Eigen::Matrix<T, 2, 2> expected_x{{T(2), T(0)}, {T(0), T(0)}};

  auto check = [&](slp::Variable<T> f, T x0, T y0, T x1, T y1) {
    slp::Hessian hessian{f, input};
    auto symbolic_H = hessian.get();

    input[0].set_value(x0);
    input[1].set_value(y0);
    CHECK(symbolic_H.value() == expected_y);
    CHECK(hessian.value().toDense() == expected_y);

    input[0].set_value(x1);
    input[1].set_value(y1);
    CHECK(symbolic_H.value() == expected_x);
    CHECK(hessian.value().toDense() == expected_x);

    input[0].set_value(T(2));
    input[1].set_value(T(2));
    CHECK(symbolic_H.value() == expected_x);
    CHECK(hessian.value().toDense() == expected_x);
  };

  auto x_squared = input[0] * input[0];
  auto y_squared = input[1] * input[1];
  check(slp::max(x_squared, y_squared), T(1), T(2), T(3), T(2));
  check(slp::min(x_squared, y_squared), T(2), T(1), T(2), T(3));
}

TEMPLATE_TEST_CASE("Hessian - pow()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::log;
  using std::pow;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> input{2};
  auto f = slp::pow(input[0] + T(1), input[1]);
  slp::Hessian hessian{f, input};

  auto symbolic_H = hessian.get();

  input[0].set_value(T(2));
  input[1].set_value(T(3));
  T mixed = T(9) * (T(1) + T(3) * log(T(3)));
  Eigen::Matrix<T, 2, 2> expected{{T(18), mixed},
                                  {mixed, T(27) * pow(log(T(3)), T(2))}};
  CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
  CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));

  input[0].set_value(T(3));
  input[1].set_value(T(2));
  mixed = T(4) * (T(1) + T(2) * log(T(4)));
  expected = Eigen::Matrix<T, 2, 2>{{T(2), mixed},
                                    {mixed, T(16) * pow(log(T(4)), T(2))}};
  CHECK_THAT(symbolic_H.value(), WithinAbs(expected, T(1e-12)));
  CHECK_THAT(hessian.value().toDense(), WithinAbs(expected, T(1e-12)));
}

TEMPLATE_TEST_CASE("Hessian - sign()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::sign(u); },
                         [](T) { return T(0); }, [](T) { return T(0); });
}

TEMPLATE_TEST_CASE("Hessian - sin()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cos;
  using std::sin;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::sin(u); },
                         [](T u) { return cos(u); },
                         [](T u) { return -sin(u); });
}

TEMPLATE_TEST_CASE("Hessian - sinh()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cosh;
  using std::sinh;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::sinh(u); },
                         [](T u) { return cosh(u); },
                         [](T u) { return sinh(u); });
}

TEMPLATE_TEST_CASE("Hessian - sqrt()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::sqrt;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::sqrt(u); },
                         [](T u) { return T(1) / (T(2) * sqrt(u)); },
                         [](T u) { return T(-1) / (T(4) * u * sqrt(u)); });
}

TEMPLATE_TEST_CASE("Hessian - tan()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cos;
  using std::tan;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>([](const auto& u) { return slp::tan(u); },
                         [](T u) { return T(1) / (cos(u) * cos(u)); },
                         [](T u) { return T(2) * tan(u) / (cos(u) * cos(u)); });
}

TEMPLATE_TEST_CASE("Hessian - tanh()", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cosh;
  using std::tanh;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  check_unary_hessian<T>(
      [](const auto& u) { return slp::tanh(u); },
      [](T u) { return T(1) / (cosh(u) * cosh(u)); },
      [](T u) { return T(-2) * tanh(u) / (cosh(u) * cosh(u)); });
}

TEMPLATE_TEST_CASE("Hessian - Rosenbrock", "[Hessian]", SCALAR_TYPES) {
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
                   SCALAR_TYPES) {
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
  CHECK(J.get().value() == expected_J);
  CHECK(J.value().toDense() == expected_J);

  //           [ 2sin(x₁)    2x₀cos(x₁)]
  // d²y/dx² = [2x₀cos(x₁)  −x₀²sin(x₁)]
  //
  //           [2sin(4)   6cos(4)]
  // d²y/dx² = [6cos(4)  −9sin(4)]
  auto H = slp::Hessian(y, x);
  Eigen::Matrix<T, 2, 2> expected_H{{T(2) * sin(T(4)), T(6) * cos(T(4))},
                                    {T(6) * cos(T(4)), T(-9) * sin(T(4))}};
  CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEMPLATE_TEST_CASE("Hessian - Edge pushing (Petro) figure 1", "[Hessian]",
                   SCALAR_TYPES) {
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

TEMPLATE_TEST_CASE("Hessian - Linear node elimination", "[Hessian]",
                   SCALAR_TYPES) {
  using T = TestType;
  using std::cos;
  using std::exp;
  using std::sin;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{3};

  // Each term exercises a different elimination case:
  //
  //   sin(x₀ + x₁ − x₂): chain of sums with one parent each
  //   exp(2x₁) cos(2x₁): constant multiple with two parents
  //   (x₀ + x₂)²: sum with two parents and two children (not eliminated)
  //   (3x₂)²: eliminated node as both arguments of a product
  //   exp(x₀ − x₀ + x₁): expansion terms that cancel
  //   log(x₁ + x₀/4): quotient by a constant
  slp::Variable two_x1 = T(2) * x[1];
  slp::Variable sum = x[0] + x[2];
  slp::Variable three_x2 = T(3) * x[2];
  auto f = slp::sin(x[0] + x[1] - x[2]) + slp::exp(two_x1) * slp::cos(two_x1) +
           sum * sum + three_x2 * three_x2 + slp::exp(x[0] - x[0] + x[1]) +
           slp::log(x[1] + x[0] / T(4));

  slp::Hessian<T, Eigen::Lower | Eigen::Upper> H{f, x};

  for (const auto& point : {Eigen::Vector<T, 3>{T(1), T(2), T(3)},
                            Eigen::Vector<T, 3>{T(-0.5), T(0.25), T(1.5)}}) {
    x.set_value(point);
    const T x0 = point[0];
    const T x1 = point[1];
    const T x2 = point[2];

    Eigen::Vector<T, 3> v{T(1), T(1), T(-1)};
    Eigen::Vector<T, 3> w{T(0.25), T(1), T(0)};
    const T s = x1 + x0 / T(4);

    Eigen::Matrix<T, 3, 3> expected_H =
        -sin(x0 + x1 - x2) * v * v.transpose() - w * w.transpose() / (s * s);
    expected_H(1, 1) += T(-8) * exp(T(2) * x1) * sin(T(2) * x1) + exp(x1);
    expected_H(0, 0) += T(2);
    expected_H(0, 2) += T(2);
    expected_H(2, 0) += T(2);
    expected_H(2, 2) += T(2) + T(18);

    CHECK_THAT(H.value().toDense(), WithinAbs(expected_H, T(1e-12)));
    CHECK_THAT(H.get().value(), WithinAbs(expected_H, T(1e-12)));
  }
}

TEMPLATE_TEST_CASE("Hessian - Copy", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;
  using std::cos;
  using std::sin;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{2};
  x[0].set_value(T(1));
  x[1].set_value(T(2));

  // f = sin(x₀x₁) + x₀²x₁
  //
  // sin(x₀x₁) stores temporary Hessian entries, which a copied Hessian has its
  // own copies of
  auto f = slp::sin(x[0] * x[1]) + x[0] * x[0] * x[1];

  auto expected_H = [&] {
    const T x0 = x[0].value();
    const T x1 = x[1].value();
    const T s = sin(x0 * x1);
    const T c = cos(x0 * x1);
    return Eigen::Matrix<T, 2, 2>{
        {-x1 * x1 * s + T(2) * x1, c - x0 * x1 * s + T(2) * x0},
        {c - x0 * x1 * s + T(2) * x0, -x0 * x0 * s}};
  };

  slp::Hessian<T, Eigen::Lower | Eigen::Upper> H{f, x};
  CHECK_THAT(H.value().toDense(), WithinAbs(expected_H(), T(1e-12)));

  auto H_copy = H;
  x[0].set_value(T(-0.5));
  x[1].set_value(T(3));
  CHECK_THAT(H_copy.value().toDense(), WithinAbs(expected_H(), T(1e-12)));
  CHECK_THAT(H.value().toDense(), WithinAbs(expected_H(), T(1e-12)));

  H = H_copy;
  x[0].set_value(T(0.25));
  CHECK_THAT(H.value().toDense(), WithinAbs(expected_H(), T(1e-12)));
}

TEMPLATE_TEST_CASE("Hessian - Constant contributions", "[Hessian]",
                   SCALAR_TYPES) {
  using T = TestType;
  using std::sin;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{2};

  // f = x₀x₁ + 3x₁² − x₀x₀ + sin(x₀)
  //
  // The products have constant adjoints and second partial derivatives, so
  // their contributions are constant and folded into the results's initial
  // values. sin(x₀) isn't, so it's recomputed on each evaluation.
  auto f = x[0] * x[1] + T(3) * x[1] * x[1] - x[0] * x[0] + slp::sin(x[0]);

  slp::Hessian<T, Eigen::Lower | Eigen::Upper> H{f, x};
  auto symbolic_H = H.get();

  for (T x0 : {T(1), T(-2), T(0.5)}) {
    x[0].set_value(x0);
    x[1].set_value(T(2) * x0);

    Eigen::Matrix<T, 2, 2> expected_H{{T(-2) - sin(x0), T(1)}, {T(1), T(6)}};
    CHECK_THAT(H.value().toDense(), WithinAbs(expected_H, T(1e-12)));
    CHECK_THAT(symbolic_H.value(), WithinAbs(expected_H, T(1e-12)));
  }
}

TEMPLATE_TEST_CASE("Hessian - Lower triangle", "[Hessian]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{3};

  // f = x₀²x₁ + (x₀ − 2x₁)², with x₂ in none of the terms
  //
  //     [2x₁ + 2     2x₀ − 4  0]
  // H = [2x₀ − 4        8     0]
  //     [   0           0     0]
  auto d = x[0] - T(2) * x[1];
  auto f = x[0] * x[0] * x[1] + d * d;

  slp::Hessian<T, Eigen::Lower> H{f, x};
  auto symbolic_H = H.get();

  for (auto [x0, x1] : {std::pair{T(1), T(2)}, std::pair{T(3), T(-1)}}) {
    x[0].set_value(x0);
    x[1].set_value(x1);

    Eigen::Matrix<T, 3, 3> expected_H{{T(2) * x1 + T(2), T(0), T(0)},
                                      {T(2) * x0 - T(4), T(8), T(0)},
                                      {T(0), T(0), T(0)}};
    CHECK(Eigen::Matrix<T, 3, 3>{H.value()} == expected_H);

    // get() only sets the lower triangle
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col <= row; ++col) {
        CHECK(symbolic_H[row, col].value() == expected_H(row, col));
      }
    }
  }
}

TEMPLATE_TEST_CASE("Hessian - pow() with power of 0.5", "[Hessian]",
                   SCALAR_TYPES) {
  using T = TestType;
  using std::pow;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{1};
  x[0].set_value(T(2));

  // pow(x, 0.5) is rewritten as sqrt(x)
  auto y = slp::pow(x[0], T(0.5));
  CHECK_THAT(y.value(), WithinAbs(pow(T(2), T(0.5)), T(1e-15)));

  // d²/dx² √x = −1/4 x^(−3/2)
  auto H = slp::Hessian(y, x);
  CHECK_THAT(H.value().coeff(0, 0),
             WithinAbs(T(-0.25) * pow(T(2), T(-1.5)), T(1e-15)));
}

TEMPLATE_TEST_CASE("Hessian - Variable reuse", "[Hessian]", SCALAR_TYPES) {
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
