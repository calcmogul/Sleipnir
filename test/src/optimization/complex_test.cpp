// Copyright (c) Sleipnir contributors

#include <cmath>
#include <complex>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

namespace {

template <typename T>
bool near(const std::complex<T>& actual, const std::complex<T>& expected) {
  using std::abs;
  constexpr T tolerance(1e-6);
  return abs(actual.real() - expected.real()) <= tolerance &&
         abs(actual.imag() - expected.imag()) <= tolerance;
}

template <typename T>
bool near(const Eigen::MatrixX<std::complex<T>>& actual,
          const Eigen::MatrixX<std::complex<T>>& expected) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
    return false;
  }
  for (int row = 0; row < actual.rows(); ++row) {
    for (int col = 0; col < actual.cols(); ++col) {
      if (!near<T>(actual[row, col], expected[row, col])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

TEMPLATE_TEST_CASE("Complex - Arithmetic", "[Complex]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;

  auto z = problem.complex_decision_variable();
  CHECK(z.value() == C{T(0), T(0)});

  z.set_value(C{T(1), T(2)});
  const C w{T(3), T(-1)};
  const C z_value{T(1), T(2)};

  // (1 + 2i)(3 − i) = 5 + 5i
  CHECK(near<T>((z * w).value(), C{T(5), T(5)}));
  CHECK(near<T>((w * z).value(), C{T(5), T(5)}));
  // (1 + 2i)/(3 − i) = (1 + 7i)/10
  CHECK(near<T>((z / w).value(), C{T(0.1), T(0.7)}));
  // (3 − i)/(1 + 2i) = (1 − 7i)/5
  CHECK(near<T>((w / z).value(), C{T(0.2), T(-1.4)}));
  CHECK(near<T>((z / z).value(), C{T(1), T(0)}));
  CHECK(near<T>((z + w).value(), C{T(4), T(1)}));
  CHECK(near<T>((z - w).value(), C{T(-2), T(3)}));
  CHECK(near<T>((w - z).value(), C{T(2), T(-3)}));
  CHECK(near<T>((-z).value(), -z_value));
  CHECK(near<T>((z * T(2)).value(), C{T(2), T(4)}));
  CHECK(near<T>((T(2) * z).value(), C{T(2), T(4)}));
  CHECK(near<T>((z / T(2)).value(), C{T(0.5), T(1)}));
  CHECK(near<T>((z * z.real()).value(), C{T(1), T(2)}));
  CHECK(near<T>(z.conj().value(), C{T(1), T(-2)}));

  constexpr T tolerance(1e-6);
  CHECK_THAT(z.abs2().value(), WithinAbs(T(5), tolerance));
  CHECK_THAT(z.abs().value(), WithinAbs(T(std::sqrt(5.0)), tolerance));
  CHECK_THAT(z.arg().value(), WithinAbs(T(std::atan2(2.0, 1.0)), tolerance));

  // Real × constant complex
  auto x = problem.real_decision_variable();
  x.set_value(T(2));
  CHECK(near<T>((x * w).value(), C{T(6), T(-2)}));
  CHECK(near<T>((w * x).value(), C{T(6), T(-2)}));
  CHECK(near<T>((x + w).value(), C{T(5), T(-1)}));
  CHECK(near<T>((x - w).value(), C{T(-1), T(1)}));
  CHECK(near<T>((x / w).value(), C{T(0.6), T(0.2)}));
  CHECK(near<T>((w / x).value(), C{T(1.5), T(-0.5)}));

  auto p = slp::Complex<T>::polar(x, T(std::numbers::pi / 2));
  CHECK(near<T>(p.value(), C{T(0), T(2)}));
}

TEMPLATE_TEST_CASE("Complex - Gradient", "[Complex]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;

  auto z = problem.complex_decision_variable();
  z.set_value(C{T(1), T(2)});
  const C w{T(3), T(-1)};

  // Re((a + bi)(c + di)) = ac − bd
  auto f = (z * w).real();
  CHECK(slp::Gradient(f, z.real()).value().coeff(0) == T(3));
  CHECK(slp::Gradient(f, z.imag()).value().coeff(0) == T(1));

  // |z|² = a² + b²
  auto g = z.abs2();
  CHECK(slp::Gradient(g, z.real()).value().coeff(0) == T(2));
  CHECK(slp::Gradient(g, z.imag()).value().coeff(0) == T(4));
}

TEMPLATE_TEST_CASE("Complex - Equality constraints", "[Complex]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;

  auto z = problem.complex_decision_variable();
  z.set_value(C{T(1), T(2)});

  auto eq = z == C{T(1), T(2)};
  CHECK(eq.constraints.size() == 2);
  CHECK(static_cast<bool>(eq));
  CHECK_FALSE(static_cast<bool>(C{T(1), T(3)} == z));
}

TEMPLATE_TEST_CASE("ComplexMatrix - Transposes and parts", "[ComplexMatrix]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;

  auto Z = problem.complex_decision_variable(2, 3);
  CHECK(Z.rows() == 2);
  CHECK(Z.cols() == 3);

  Eigen::MatrixX<C> value{{C{T(1), T(2)}, C{T(3), T(-1)}, C{T(0), T(1)}},
                          {C{T(-2), T(0)}, C{T(1), T(1)}, C{T(4), T(-3)}}};
  Z.set_value(value);

  CHECK(near<T>(Z.value(), value));
  CHECK(near<T>(Z.T().value(), Eigen::MatrixX<C>{value.transpose()}));
  CHECK(near<T>(Z.conj().value(), Eigen::MatrixX<C>{value.conjugate()}));
  CHECK(near<T>(Z.H().value(), Eigen::MatrixX<C>{value.adjoint()}));
  CHECK(Z.real().value() == value.real());
  CHECK(Z.imag().value() == value.imag());
  CHECK(Z.abs2().value() == value.cwiseAbs2());

  auto row = Z.row(1);
  CHECK(near<T>(row.value(), Eigen::MatrixX<C>{value.row(1)}));
  auto col = Z.col(2);
  CHECK(near<T>(col.value(), Eigen::MatrixX<C>{value.col(2)}));
  auto block = Z.block(0, 1, 2, 2);
  CHECK(near<T>(block.value(), Eigen::MatrixX<C>{value.block(0, 1, 2, 2)}));
  auto slice = Z[slp::Slice{0, 2}, slp::Slice{0, 3, 2}];
  CHECK(slice.rows() == 2);
  CHECK(slice.cols() == 2);
  CHECK(near<T>(slice.value(0, 1), value[0, 2]));
}

TEMPLATE_TEST_CASE("ComplexMatrix - Slices", "[ComplexMatrix]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;

  auto Z = problem.complex_decision_variable(2, 3);
  Eigen::MatrixX<C> value{{C{T(1), T(2)}, C{T(3), T(-1)}, C{T(0), T(1)}},
                          {C{T(-2), T(0)}, C{T(1), T(1)}, C{T(4), T(-3)}}};
  Z.set_value(value);

  // Slices point to the matrix's entries
  auto block = Z.block(0, 1, 2, 2);
  block.set_value(Eigen::MatrixX<C>{{C{T(5), T(0)}, C{T(6), T(0)}},
                                    {C{T(7), T(0)}, C{T(8), T(1)}}});
  CHECK(Z.value(0, 1) == C{T(5), T(0)});
  CHECK(Z.value(1, 2) == C{T(8), T(1)});

  auto slice = Z[slp::Slice{0, 2}, slp::Slice{0, 3, 2}];
  CHECK(slice.rows() == 2);
  CHECK(slice.cols() == 2);
  CHECK(slice.value(1, 1) == Z.value(1, 2));
  CHECK(slice.col(1).value(0) == Z.value(0, 2));

  // Assignment writes through to the matrix
  Z.row(1) = Eigen::MatrixX<C>{{C{T(0), T(1)}, C{T(0), T(2)}, C{T(0), T(3)}}};
  CHECK(Z.value(1, 0) == C{T(0), T(1)});
  CHECK(Z.value(1, 2) == C{T(0), T(3)});

  Z.col(0) = Z.col(2);
  CHECK(Z.value(0, 0) == Z.value(0, 2));
  CHECK(Z.value(1, 0) == Z.value(1, 2));

  Z[0, 1] = C{T(2), T(-2)};
  Z.block(0, 1, 1, 1) *= C{T(0), T(1)};
  CHECK(near<T>(Z.value(0, 1), C{T(2), T(2)}));

  Z.row(0) += slp::ComplexMatrix<T>::identity(3).row(0);
  CHECK(near<T>(Z.value(0, 0), C{T(1), T(0)} + Z.value(0, 2)));

  // Slices are operands of complex matrix arithmetic
  slp::ComplexMatrix<T> product = Z.row(0) * Z.row(1).T();
  C expected{};
  for (int i = 0; i < 3; ++i) {
    expected += Z.value(0, i) * Z.value(1, i);
  }
  CHECK(product.rows() == 1);
  CHECK(product.cols() == 1);
  CHECK(near<T>(product.value(0, 0), expected));

  CHECK(near<T>(Z.col(1).H().value(),
                Eigen::MatrixX<C>{Z.value().col(1).adjoint()}));
  CHECK(Z.row(1).real().value() == Z.value().row(1).real());
  CHECK(Z.row(1).imag().value() == Z.value().row(1).imag());

  const auto& Z_const = Z;
  CHECK(Z_const.segment(0, 1).rows() == 1);
  CHECK(near<T>(Z_const.block(0, 0, 2, 2).trace().value(),
                Z.value(0, 0) + Z.value(1, 1)));

  auto constraints = Z.col(0) == Z.col(2);
  CHECK(constraints.constraints.size() == 4);
}

TEMPLATE_TEST_CASE("ComplexMatrix - Arithmetic", "[ComplexMatrix]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;

  auto V = problem.complex_decision_variable(3);

  Eigen::VectorX<C> v{{C{T(1), T(0)}, C{T(0.9), T(-0.1)}, C{T(1.05), T(0.2)}}};
  V.set_value(v);

  Eigen::MatrixX<C> Y{{C{T(2), T(-20)}, C{T(-1), T(10)}, C{T(-1), T(10)}},
                      {C{T(-1), T(10)}, C{T(1), T(-10)}, C{T(0), T(0)}},
                      {C{T(-1), T(10)}, C{T(0), T(0)}, C{T(1), T(-10)}}};
  Eigen::SparseMatrix<C> Y_sparse = Y.sparseView();

  Eigen::VectorX<C> Yv = Y * v;

  CHECK(near<T>((Y * V).value(), Eigen::MatrixX<C>{Yv}));
  CHECK(near<T>((Y_sparse * V).value(), Eigen::MatrixX<C>{Yv}));
  CHECK(near<T>((V.T() * Y).value(), Eigen::MatrixX<C>{v.transpose() * Y}));
  CHECK(near<T>((V.T() * Y_sparse).value(),
                Eigen::MatrixX<C>{v.transpose() * Y}));

  // Complex power injection S = V ∘ conj(Y V)
  Eigen::VectorX<C> S = v.cwiseProduct(Yv.conjugate());
  CHECK(near<T>(V.cwise_product((Y * V).conj()).value(), Eigen::MatrixX<C>{S}));

  // Hermitian form Vᴴ Y V
  C vhyv = (v.adjoint() * Y * v)(0, 0);
  CHECK(near<T>(slp::Complex<T>{V.H() * Y * V}.value(), vhyv));

  // Real matrices mix with complex ones
  Eigen::MatrixX<T> A{
      {T(1), T(2), T(0)}, {T(0), T(1), T(-1)}, {T(3), T(0), T(1)}};
  slp::RealMatrix<T> A_var{A};
  Eigen::MatrixX<C> Av = (A.template cast<C>() * v).eval();
  CHECK(near<T>((A * V).value(), Av));
  CHECK(near<T>((A_var * V).value(), Av));
  CHECK(near<T>((A_var * Y).value(),
                Eigen::MatrixX<C>{A.template cast<C>() * Y}));

  CHECK(near<T>((V + V).value(), Eigen::MatrixX<C>{v + v}));
  CHECK(near<T>((V - Yv).value(), Eigen::MatrixX<C>{v - Yv}));
  CHECK(near<T>((Yv - V).value(), Eigen::MatrixX<C>{Yv - v}));
  CHECK(near<T>((-V).value(), Eigen::MatrixX<C>{-v}));

  const C w{T(3), T(-1)};
  CHECK(near<T>((V * w).value(), Eigen::MatrixX<C>{v * w}));
  CHECK(near<T>((w * V).value(), Eigen::MatrixX<C>{w * v}));
  CHECK(near<T>((V / w).value(), Eigen::MatrixX<C>{v / w}));
  CHECK(near<T>((V * T(2)).value(), Eigen::MatrixX<C>{v * T(2)}));
  CHECK(near<T>((A_var * w).value(),
                Eigen::MatrixX<C>{A.template cast<C>() * w}));

  auto z = problem.complex_decision_variable();
  z.set_value(w);
  CHECK(near<T>((V * z).value(), Eigen::MatrixX<C>{v * w}));
  CHECK(near<T>((z * V).value(), Eigen::MatrixX<C>{w * v}));

  slp::ComplexMatrix<T> W = V;
  W += Yv;
  W -= V;
  W *= T(2);
  W /= w;
  CHECK(near<T>(W.value(), Eigen::MatrixX<C>{T(2) * Yv / w}));

  CHECK(near<T>(slp::ComplexMatrix<T>{Y * V * V.T()}.trace().value(),
                (Y * v * v.transpose()).trace()));
  CHECK(near<T>(slp::ComplexMatrix<T>::identity(2).value(),
                Eigen::MatrixX<C>::Identity(2, 2)));
}

TEMPLATE_TEST_CASE("ComplexMatrix - Equality constraints", "[ComplexMatrix]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;

  auto V = problem.complex_decision_variable(2);

  Eigen::VectorX<C> v{{C{T(1), T(-1)}, C{T(2), T(0.5)}}};
  V.set_value(v);

  auto eq = V == v;
  CHECK(eq.constraints.size() == 4);
  CHECK(static_cast<bool>(eq));
  CHECK((v == V).constraints.size() == 4);
  CHECK((V == C{T(0), T(0)}).constraints.size() == 4);
  CHECK((V == V.conj().conj()).constraints.size() == 4);
}

TEMPLATE_TEST_CASE("Complex - Parameters", "[Complex]",
                   FLOATING_POINT_SCALAR_TYPES) {
  using T = TestType;
  using C = std::complex<T>;

  slp::Problem<T> problem;
  auto z = problem.complex_parameter();
  auto Z = problem.complex_parameter(2);
  CHECK(z.value() == C{T(0), T(0)});
  CHECK(Z.value() == Eigen::MatrixX<C>::Zero(2, 1));

  z.set_value(C{T(1), T(-2)});
  CHECK(z.value() == C{T(1), T(-2)});
}
