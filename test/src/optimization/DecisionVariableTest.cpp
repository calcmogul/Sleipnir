// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("DecisionVariable - Scalar init assign", "[DecisionVariable]") {
  sleipnir::OptimizationProblem problem;

  // Scalar zero init
  auto x = problem.DecisionVariable();
  CHECK(x.Value() == 0.0);

  // Scalar assignment
  x.SetValue(1.0);
  CHECK(x.Value() == 1.0);
  x.SetValue(2.0);
  CHECK(x.Value() == 2.0);
}

TEST_CASE("DecisionVariable - Vector init assign", "[DecisionVariable]") {
  sleipnir::OptimizationProblem problem;

  // Vector zero init
  auto y = problem.DecisionVariable(2);
  CHECK(y.Value(0) == 0.0);
  CHECK(y.Value(1) == 0.0);

  // Vector assignment
  y[0].SetValue(1.0);
  y[1].SetValue(2.0);
  CHECK(y.Value(0) == 1.0);
  CHECK(y.Value(1) == 2.0);
  y[0].SetValue(3.0);
  y[1].SetValue(4.0);
  CHECK(y.Value(0) == 3.0);
  CHECK(y.Value(1) == 4.0);
}

TEST_CASE("DecisionVariable - Static matrix init assign",
          "[DecisionVariable]") {
  sleipnir::OptimizationProblem problem;

  // Matrix zero init
  auto z = problem.DecisionVariable(3, 2);
  CHECK(z.Value(0, 0) == 0.0);
  CHECK(z.Value(0, 1) == 0.0);
  CHECK(z.Value(1, 0) == 0.0);
  CHECK(z.Value(1, 1) == 0.0);
  CHECK(z.Value(2, 0) == 0.0);
  CHECK(z.Value(2, 1) == 0.0);

  // Matrix assignment; element comparison
  z.SetValue(Eigen::Matrix<double, 3, 2>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  CHECK(z.Value(0, 0) == 1.0);
  CHECK(z.Value(0, 1) == 2.0);
  CHECK(z.Value(1, 0) == 3.0);
  CHECK(z.Value(1, 1) == 4.0);
  CHECK(z.Value(2, 0) == 5.0);
  CHECK(z.Value(2, 1) == 6.0);

  // Matrix assignment; matrix comparison
  {
    Eigen::Matrix<double, 3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    z.SetValue(expected);
    CHECK(z.Value() == expected);
  }

  // Block assignment
  {
    Eigen::Matrix<double, 2, 1> expectedBlock{{1.0}, {1.0}};
    z.Block(0, 0, 2, 1).SetValue(expectedBlock);

    Eigen::Matrix<double, 3, 2> expectedResult{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    CHECK(z.Value() == expectedResult);
  }

  // Segment assignment
  {
    Eigen::Matrix<double, 2, 1> expectedBlock{{1.0}, {1.0}};
    z.Segment(0, 2).SetValue(expectedBlock);

    Eigen::Matrix<double, 3, 2> expectedResult{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    CHECK(z.Value() == expectedResult);
  }
}

TEST_CASE("DecisionVariable - Dynamic matrix init assign",
          "[DecisionVariable]") {
  sleipnir::OptimizationProblem problem;

  // Matrix zero init
  auto z = problem.DecisionVariable(3, 2);
  CHECK(z.Value(0, 0) == 0.0);
  CHECK(z.Value(0, 1) == 0.0);
  CHECK(z.Value(1, 0) == 0.0);
  CHECK(z.Value(1, 1) == 0.0);
  CHECK(z.Value(2, 0) == 0.0);
  CHECK(z.Value(2, 1) == 0.0);

  // Matrix assignment; element comparison
  {
    Eigen::MatrixXd expected{3, 2};
    expected << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    z.SetValue(expected);
    CHECK(z.Value(0, 0) == 1.0);
    CHECK(z.Value(0, 1) == 2.0);
    CHECK(z.Value(1, 0) == 3.0);
    CHECK(z.Value(1, 1) == 4.0);
    CHECK(z.Value(2, 0) == 5.0);
    CHECK(z.Value(2, 1) == 6.0);
  }

  // Matrix assignment; matrix comparison
  {
    Eigen::MatrixXd expected{3, 2};
    expected << 7.0, 8.0, 9.0, 10.0, 11.0, 12.0;
    z.SetValue(expected);
    CHECK(z.Value() == expected);
  }

  // Block assignment
  {
    Eigen::MatrixXd expectedBlock{2, 1};
    expectedBlock << 1.0, 1.0;
    z.Block(0, 0, 2, 1).SetValue(expectedBlock);

    Eigen::MatrixXd expectedResult{3, 2};
    expectedResult << 1.0, 8.0, 1.0, 10.0, 11.0, 12.0;
    CHECK(z.Value() == expectedResult);
  }

  // Segment assignment
  {
    Eigen::MatrixXd expectedBlock{2, 1};
    expectedBlock << 1.0, 1.0;
    z.Segment(0, 2).SetValue(expectedBlock);

    Eigen::MatrixXd expectedResult{3, 2};
    expectedResult << 1.0, 8.0, 1.0, 10.0, 11.0, 12.0;
    CHECK(z.Value() == expectedResult);
  }
}

TEST_CASE("DecisionVariable - Symmetric matrix", "[DecisionVariable]") {
  sleipnir::OptimizationProblem problem;

  // Matrix zero init
  auto A = problem.SymmetricDecisionVariable(2);
  CHECK(A.Value(0, 0) == 0.0);
  CHECK(A.Value(0, 1) == 0.0);
  CHECK(A.Value(1, 0) == 0.0);
  CHECK(A.Value(1, 1) == 0.0);

  // Assign to lower triangle
  A[0, 0].SetValue(1.0);
  A[1, 0].SetValue(2.0);
  A[1, 1].SetValue(3.0);

  // Confirm whole matrix changed
  CHECK(A.Value(0, 0) == 1.0);
  CHECK(A.Value(0, 1) == 2.0);
  CHECK(A.Value(1, 0) == 2.0);
  CHECK(A.Value(1, 1) == 3.0);
}
