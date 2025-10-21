// Copyright (c) Sleipnir contributors

#include <array>
#include <tuple>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("constraints - Equality constraint boolean comparison",
                   "[constraints]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using slp::Variable;
  using slp::VariableMatrix;
  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  constexpr std::array args{std::tuple{T(1.0), T(1.0)},
                            std::tuple{T(1.0), T(2.0)},
                            std::tuple{T(2.0), T(1.0)}};

  // T-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} == Variable<T>{rhs}} == (lhs == rhs));
  }

  // T-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} == VariableMatrix<T>{{rhs}}} == (lhs == rhs));
  }

  // Variable-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} == T{rhs}} == (lhs == rhs));
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} == Variable<T>{rhs}} == (lhs == rhs));
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} == VariableMatrix<T>{{rhs}}} == (lhs == rhs));
  }

  // VariableMatrix-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} == T{rhs}} == (lhs == rhs));
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} == Variable<T>{rhs}} == (lhs == rhs));
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} == VariableMatrix<T>{{rhs}}} ==
          (lhs == rhs));
  }

  // MatrixXT-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{MatrixXT{{lhs}} == Variable<T>{rhs}} == (lhs == rhs));
  }

  // MatrixXT-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{MatrixXT{{lhs}} == VariableMatrix<T>{{rhs}}} == (lhs == rhs));
  }

  // MatrixXT-VariableBlock
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{MatrixXT{{lhs}} == VariableMatrix<T>{{rhs}}.block(0, 0, 1, 1)} ==
          (lhs == rhs));
  }

  // Variable-MatrixXT
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} == MatrixXT{{rhs}}} == (lhs == rhs));
  }

  // VariableMatrix-MatrixXT
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} == MatrixXT{{rhs}}} == (lhs == rhs));
  }

  // VariableBlock-MatrixXT
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}}.block(0, 0, 1, 1) == MatrixXT{{rhs}}} ==
          (lhs == rhs));
  }
}

// For the purposes of optimization, a < constraint is treated the same as a <=
// constraint
TEMPLATE_TEST_CASE("constraints - Inequality constraint boolean comparisons",
                   "[constraints]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using slp::Variable;
  using slp::VariableMatrix;
  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  constexpr std::array args{std::tuple{T(1.0), T(1.0)},
                            std::tuple{T(1.0), T(2.0)},
                            std::tuple{T(2.0), T(1.0)}};

  // T-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} < Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{T{lhs} <= Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{T{lhs} > Variable<T>{rhs}} == (lhs >= rhs));
    CHECK(bool{T{lhs} >= Variable<T>{rhs}} == (lhs >= rhs));
  }

  // T-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} < VariableMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{T{lhs} <= VariableMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{T{lhs} > VariableMatrix<T>{{rhs}}} == (lhs >= rhs));
    CHECK(bool{T{lhs} >= VariableMatrix<T>{{rhs}}} == (lhs >= rhs));
  }

  // Variable-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} < T{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} <= T{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} > T{rhs}} == (lhs >= rhs));
    CHECK(bool{Variable<T>{lhs} >= T{rhs}} == (lhs >= rhs));
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} < Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} <= Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} > Variable<T>{rhs}} == (lhs >= rhs));
    CHECK(bool{Variable<T>{lhs} >= Variable<T>{rhs}} == (lhs >= rhs));
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} < VariableMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} <= VariableMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} > VariableMatrix<T>{{rhs}}} == (lhs >= rhs));
    CHECK(bool{Variable<T>{lhs} >= VariableMatrix<T>{{rhs}}} == (lhs >= rhs));
  }

  // VariableMatrix-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} < T{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} <= T{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} > T{rhs}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} >= T{rhs}} == (lhs >= rhs));
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} < Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} <= Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} > Variable<T>{rhs}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} >= Variable<T>{rhs}} == (lhs >= rhs));
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} < VariableMatrix<T>{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} <= VariableMatrix<T>{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} > VariableMatrix<T>{{rhs}}} ==
          (lhs >= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} >= VariableMatrix<T>{{rhs}}} ==
          (lhs >= rhs));
  }

  // MatrixXT-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{MatrixXT{{lhs}} < Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{MatrixXT{{lhs}} <= Variable<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{MatrixXT{{lhs}} > Variable<T>{rhs}} == (lhs >= rhs));
    CHECK(bool{MatrixXT{{lhs}} >= Variable<T>{rhs}} == (lhs >= rhs));
  }

  // MatrixXT-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{MatrixXT{{lhs}} < VariableMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{MatrixXT{{lhs}} <= VariableMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{MatrixXT{{lhs}} > VariableMatrix<T>{{rhs}}} == (lhs >= rhs));
    CHECK(bool{MatrixXT{{lhs}} >= VariableMatrix<T>{{rhs}}} == (lhs >= rhs));
  }

  // MatrixXT-VariableBlock
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{MatrixXT{{lhs}} < VariableMatrix<T>{{rhs}}.block(0, 0, 1, 1)} ==
          (lhs <= rhs));
    CHECK(bool{MatrixXT{{lhs}} <= VariableMatrix<T>{{rhs}}.block(0, 0, 1, 1)} ==
          (lhs <= rhs));
    CHECK(bool{MatrixXT{{lhs}} > VariableMatrix<T>{{rhs}}.block(0, 0, 1, 1)} ==
          (lhs >= rhs));
    CHECK(bool{MatrixXT{{lhs}} >= VariableMatrix<T>{{rhs}}.block(0, 0, 1, 1)} ==
          (lhs >= rhs));
  }

  // Variable-MatrixXT
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable<T>{lhs} < MatrixXT{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} <= MatrixXT{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable<T>{lhs} > MatrixXT{{rhs}}} == (lhs >= rhs));
    CHECK(bool{Variable<T>{lhs} >= MatrixXT{{rhs}}} == (lhs >= rhs));
  }

  // VariableMatrix-MatrixXT
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}} < MatrixXT{{rhs}}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} <= MatrixXT{{rhs}}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} > MatrixXT{{rhs}}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}} >= MatrixXT{{rhs}}} == (lhs >= rhs));
  }

  // VariableBlock-MatrixXT
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix<T>{{lhs}}.block(0, 0, 1, 1) < MatrixXT{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}}.block(0, 0, 1, 1) <= MatrixXT{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}}.block(0, 0, 1, 1) > MatrixXT{{rhs}}} ==
          (lhs >= rhs));
    CHECK(bool{VariableMatrix<T>{{lhs}}.block(0, 0, 1, 1) >= MatrixXT{{rhs}}} ==
          (lhs >= rhs));
  }
}

TEMPLATE_TEST_CASE("constraints - Equality constraint concatenation",
                   "[constraints]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using slp::EqualityConstraints;
  using slp::Variable;

  EqualityConstraints eq1 = Variable<T>{1.0} == Variable<T>{1.0};
  EqualityConstraints eq2 = Variable<T>{1.0} == Variable<T>{2.0};
  EqualityConstraints eqs{eq1, eq2};

  CHECK(eq1.constraints.size() == 1);
  CHECK(eq2.constraints.size() == 1);
  CHECK(eqs.constraints.size() == 2);

  CHECK(eqs.constraints[0].value() == eq1.constraints[0].value());
  CHECK(eqs.constraints[1].value() == eq2.constraints[0].value());

  CHECK(bool{eq1});
  CHECK_FALSE(bool{eq2});
  CHECK_FALSE(bool{eqs});
}

TEMPLATE_TEST_CASE("constraints - Inequality constraint concatenation",
                   "[constraints]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using slp::InequalityConstraints;
  using slp::Variable;

  InequalityConstraints ineq1 = Variable<T>{2.0} < Variable<T>{1.0};
  InequalityConstraints ineq2 = Variable<T>{1.0} < Variable<T>{2.0};
  InequalityConstraints ineqs{ineq1, ineq2};

  CHECK(ineq1.constraints.size() == 1);
  CHECK(ineq2.constraints.size() == 1);
  CHECK(ineqs.constraints.size() == 2);

  CHECK(ineqs.constraints[0].value() == ineq1.constraints[0].value());
  CHECK(ineqs.constraints[1].value() == ineq2.constraints[0].value());

  CHECK_FALSE(bool{ineq1});
  CHECK(bool{ineq2});
  CHECK_FALSE(bool{ineqs});
}
