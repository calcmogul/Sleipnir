// Copyright (c) Sleipnir contributors

#include <cmath>
#include <functional>
#include <numeric>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable.hpp>

#include "range.hpp"
#include "util/scope_exit.hpp"

TEST_CASE("Hessian - Linear", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x
  slp::VariableMatrix x{1};
  x[0].set_value(3);
  slp::Variable y = x[0];

  // dy/dx = 1
  double g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == 1.0);

  // d²y/dx² = 0
  auto H = slp::Hessian(y, x);
  // CHECK(H.get().value(0, 0) == 0.0);
  CHECK(H.value().coeff(0, 0) == 0.0);
}

TEST_CASE("Hessian - Quadratic", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x²
  slp::VariableMatrix x{1};
  x[0].set_value(3);
  slp::Variable y = x[0] * x[0];

  // dy/dx = 2x = 6
  double g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == 6.0);

  // d²y/dx² = 2
  auto H = slp::Hessian(y, x);
  // CHECK(H.get().value(0, 0) == 2.0);
  CHECK(H.value().coeff(0, 0) == 2.0);
}

TEST_CASE("Hessian - Cubic", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x³
  slp::VariableMatrix x{1};
  x[0].set_value(3);
  slp::Variable y = x[0] * x[0] * x[0];

  // dy/dx = 3x² = 27
  double g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == 27.0);

  // d²y/dx² = 6x = 18
  auto H = slp::Hessian(y, x);
  // CHECK(H.get().value(0, 0) == 18.0);
  CHECK(H.value().coeff(0, 0) == 18.0);
}

TEST_CASE("Hessian - Quartic", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x⁴
  slp::VariableMatrix x{1};
  x[0].set_value(3);
  slp::Variable y = x[0] * x[0] * x[0] * x[0];

  // dy/dx = 4x³ = 108
  double g = slp::Gradient(y, x[0]).value().coeff(0);
  CHECK(g == 108.0);

  // d²y/dx² = 12x² = 108
  auto H = slp::Hessian(y, x);
  // CHECK(H.get().value(0, 0) == 108.0);
  CHECK(H.value().coeff(0, 0) == 108.0);
}

TEST_CASE("Hessian - Sum", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{5};
  x[0].set_value(1);
  x[1].set_value(2);
  x[2].set_value(3);
  x[3].set_value(4);
  x[4].set_value(5);

  // y = sum(x)
  auto y = std::accumulate(x.begin(), x.end(), slp::Variable{0.0});
  CHECK(y.value() == 15.0);

  auto g = slp::Gradient(y, x);
  CHECK(g.get().value() == Eigen::MatrixXd::Constant(5, 1, 1.0));
  CHECK(g.value().toDense() == Eigen::MatrixXd::Constant(5, 1, 1.0));

  auto H = slp::Hessian(y, x);
  // CHECK(H.get().value() == Eigen::MatrixXd::Zero(5, 5));
  CHECK(H.value().toDense() == Eigen::MatrixXd::Zero(5, 5));
}

TEST_CASE("Hessian - Sum of products", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{5};
  for (int i = 0; i < 5; ++i) {
    x[i].set_value(i + 1);
  }

  // y = ||x||²
  slp::Variable y = x.T() * x;
  CHECK(y.value() == 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5);

  auto g = slp::Gradient(y, x);
  CHECK(g.get().value() == 2 * x.value());
  CHECK(g.value().toDense() == 2 * x.value());

  auto H = slp::Hessian(y, x);

  Eigen::MatrixXd expected_H =
      Eigen::VectorXd::Constant(5, 1, 2.0).asDiagonal();
  // CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEST_CASE("Hessian - Product of sines", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{5};
  for (int i = 0; i < 5; ++i) {
    x[i].set_value(i + 1);
  }

  // y = prod(sin(x))
  auto temp = x.cwise_transform(slp::sin);
  auto y = std::accumulate(temp.begin(), temp.end(), slp::Variable{1.0},
                           std::multiplies{});
  CHECK(y.value() == Catch::Approx(std::sin(1) * std::sin(2) * std::sin(3) *
                                   std::sin(4) * std::sin(5))
                         .margin(1e-15));

  auto g = slp::Gradient(y, x);
  for (int i = 0; i < x.rows(); ++i) {
    CHECK(g.get().value(i) ==
          Catch::Approx(y.value() / std::tan(x[i].value())).margin(1e-15));
    CHECK(g.value().coeff(i) ==
          Catch::Approx(y.value() / std::tan(x[i].value())).margin(1e-15));
  }

  auto H = slp::Hessian(y, x);

  Eigen::MatrixXd expected_H{5, 5};
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      if (i == j) {
        expected_H(i, j) = -y.value();
      } else {
        expected_H(i, j) =
            y.value() / (std::tan(x[i].value()) * std::tan(x[j].value()));
      }
    }
  }

  auto actual_H = H.get().value();
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      // CHECK(actual_H(i, j) == Catch::Approx(expected_H(i, j)).margin(1e-15));
    }
  }

  actual_H = H.value();
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      CHECK(actual_H(i, j) == Catch::Approx(expected_H(i, j)).margin(1e-15));
    }
  }
}

TEST_CASE("Hessian - Sum of squared residuals", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{5};
  for (int i = 0; i < 5; ++i) {
    x[i].set_value(1);
  }

  // y = sum(diff(x).^2)
  auto temp = (x.block(0, 0, 4, 1) - x.block(1, 0, 4, 1))
                  .cwise_transform(
                      [](const slp::Variable& x) { return slp::pow(x, 2); });
  auto y = std::accumulate(temp.begin(), temp.end(), slp::Variable{0.0});
  auto g = slp::Gradient(y, x).value().toDense();

  CHECK(y.value() == 0.0);
  CHECK(g[0] == 2 * x[0].value() - 2 * x[1].value());
  CHECK(g[1] == -2 * x[0].value() + 4 * x[1].value() - 2 * x[2].value());
  CHECK(g[2] == -2 * x[1].value() + 4 * x[2].value() - 2 * x[3].value());
  CHECK(g[3] == -2 * x[2].value() + 4 * x[3].value() - 2 * x[4].value());
  CHECK(g[4] == -2 * x[3].value() + 2 * x[4].value());

  auto H = slp::Hessian(y, x).value().toDense();

  Eigen::MatrixXd expected_H{{2.0, -2.0, 0.0, 0.0, 0.0},
                             {-2.0, 4.0, -2.0, 0.0, 0.0},
                             {0.0, -2.0, 4.0, -2.0, 0.0},
                             {0.0, 0.0, -2.0, 4.0, -2.0},
                             {0.0, 0.0, 0.0, -2.0, 2.0}};
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      CHECK(H(i, j) == expected_H(i, j));
    }
  }
}

TEST_CASE("Hessian - Sum of squares", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix r{4};
  r.set_value(Eigen::Vector<double, 4>{{25.0, 10.0, 5.0, 0.0}});

  slp::VariableMatrix x{4};
  for (int i = 0; i < 4; ++i) {
    x[i].set_value(0.0);
  }

  slp::Variable J = 0.0;
  for (int i = 0; i < 4; ++i) {
    J += (r[i] - x[i]) * (r[i] - x[i]);
  }

  auto H = slp::Hessian(J, x);

  Eigen::MatrixXd expected_H =
      Eigen::VectorXd::Constant(4, 1, 2.0).asDiagonal();
  // CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEST_CASE("Hessian - Rosenbrock", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix input{2};
  auto& x = input[0];
  auto& y = input[1];

  for (auto x0 : range(-2.5, 2.5, 0.1)) {
    for (auto y0 : range(-2.5, 2.5, 0.1)) {
      x.set_value(x0);
      y.set_value(y0);
      auto z = slp::pow(1 - x, 2) + 100 * slp::pow(y - slp::pow(x, 2), 2);

      Eigen::MatrixXd H = slp::Hessian(z, input).value();
      CHECK(H(0, 0) == Catch::Approx(-400 * (y0 - x0 * x0) + 800 * x0 * x0 + 2)
                           .margin(1e-12));
      CHECK(H(0, 1) == -400 * x0);
      CHECK(H(1, 0) == -400 * x0);
      CHECK(H(1, 1) == 200);
    }
  }
}

TEST_CASE("Hessian - Edge Pushing example 1") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{2};
  x[0].set_value(3);
  x[1].set_value(4);

  // y = (x₀sin(x₁)) x₀
  auto y = (x[0] * slp::sin(x[1])) * x[0];

  // dy/dx = [2x₀sin(x₁)  x₀²cos(x₁)]
  // dy/dx = [ 6sin(4)     9cos(4)  ]
  auto J = slp::Jacobian(y, x);
  Eigen::MatrixXd expected_J{{6.0 * std::sin(4.0), 9.0 * std::cos(4.0)}};
  // CHECK(J.get().value() == expected_J);
  CHECK(J.value().toDense() == expected_J);

  //           [ 2sin(x₁)    2x₀cos(x₁)]
  // d²y/dx² = [2x₀cos(x₁)  −x₀²sin(x₁)]
  //
  //           [2sin(4)   6cos(4)]
  // d²y/dx² = [6cos(4)  −9sin(4)]
  auto H = slp::Hessian(y, x);
  Eigen::MatrixXd expected_H{{2.0 * std::sin(4.0), 6.0 * std::cos(4.0)},
                             {6.0 * std::cos(4.0), -9.0 * std::sin(4.0)}};
  // CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEST_CASE("Hessian - Variable reuse", "[Hessian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable y;
  slp::VariableMatrix x{1};

  // y = x³
  x[0].set_value(1);
  y = x[0] * x[0] * x[0];

  slp::Hessian hessian{y, x};

  // d²y/dx² = 6x
  // H = 6
  Eigen::MatrixXd H = hessian.value();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == 6.0);

  x[0].set_value(2);
  // d²y/dx² = 6x
  // H = 12
  H = hessian.value();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == 12.0);
}
