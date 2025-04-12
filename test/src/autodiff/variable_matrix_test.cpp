// Copyright (c) Sleipnir contributors

#include <format>
#include <functional>
#include <iterator>

#include <Eigen/Core>
#include <Eigen/QR>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

TEST_CASE("VariableMatrix - Construct from Eigen::MatrixBase",
          "[VariableMatrix]") {
  slp::VariableMatrix mat{Eigen::MatrixXd{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};

  Eigen::MatrixXd expected{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  CHECK(mat.value() == expected);
}

TEST_CASE("VariableMatrix - Construct from Eigen::DiagonalBase",
          "[VariableMatrix]") {
  slp::VariableMatrix mat{Eigen::VectorXd{{1.0, 2.0, 3.0}}.asDiagonal()};

  Eigen::MatrixXd expected{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
  CHECK(mat.value() == expected);
}

TEST_CASE("VariableMatrix - Assignment to default", "[VariableMatrix]") {
  slp::VariableMatrix mat;

  CHECK(mat.rows() == 0);
  CHECK(mat.cols() == 0);

  mat = slp::VariableMatrix{2, 2};

  CHECK(mat.rows() == 2);
  CHECK(mat.cols() == 2);
  CHECK(mat(0, 0) == 0.0);
  CHECK(mat(0, 1) == 0.0);
  CHECK(mat(1, 0) == 0.0);
  CHECK(mat(1, 1) == 0.0);

  mat(0, 0) = 1.0;
  mat(0, 1) = 2.0;
  mat(1, 0) = 3.0;
  mat(1, 1) = 4.0;

  CHECK(mat(0, 0) == 1.0);
  CHECK(mat(0, 1) == 2.0);
  CHECK(mat(1, 0) == 3.0);
  CHECK(mat(1, 1) == 4.0);
}

TEST_CASE("VariableMatrix - Assignment aliasing", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0}, {3.0, 4.0}};
  slp::VariableMatrix B{{5.0, 6.0}, {7.0, 8.0}};

  Eigen::MatrixXd expected_A{{1.0, 2.0}, {3.0, 4.0}};
  Eigen::MatrixXd expected_B{{5.0, 6.0}, {7.0, 8.0}};
  CHECK(A == expected_A);
  CHECK(B == expected_B);

  A = B;

  CHECK(A == expected_B);
  CHECK(B == expected_B);

  B(0, 0).set_value(2.0);
  expected_B(0, 0) = 2.0;

  CHECK(A == expected_B);
  CHECK(B == expected_B);
}

TEST_CASE("VariableMatrix - Block() member function", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Block assignment
  A.block(1, 1, 2, 2) = Eigen::Matrix<double, 2, 2>{{10.0, 11.0}, {12.0, 13.0}};

  Eigen::Matrix<double, 3, 3> expected1{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 13.0}};
  CHECK(A.value() == expected1);

  // Block-of-block assignment
  A.block(1, 1, 2, 2).block(1, 1, 1, 1) = 14.0;

  Eigen::Matrix<double, 3, 3> expected2{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 14.0}};
  CHECK(A.value() == expected2);
}

TEST_CASE("VariableMatrix - Slicing", "[VariableMatrix]") {
  using namespace slp::slicing;

  slp::VariableMatrix mat{
      {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  CHECK(mat.rows() == 4);
  CHECK(mat.cols() == 4);

  // Single-arg index operator on full matrix
  for (int i = 0; i < mat.rows() * mat.cols(); ++i) {
    CHECK(mat[i] == i + 1);
  }

  // Slice from start
  {
    auto s = mat({1, _}, {2, _});
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 2);
    // Single-arg index operator on forward slice
    CHECK(s[0] == 7.0);
    CHECK(s[1] == 8.0);
    CHECK(s[2] == 11.0);
    CHECK(s[3] == 12.0);
    CHECK(s[4] == 15.0);
    CHECK(s[5] == 16.0);
    // Double-arg index operator on forward slice
    CHECK(s(0, 0) == 7.0);
    CHECK(s(0, 1) == 8.0);
    CHECK(s(1, 0) == 11.0);
    CHECK(s(1, 1) == 12.0);
    CHECK(s(2, 0) == 15.0);
    CHECK(s(2, 1) == 16.0);
  }

  // Slice from end
  {
    auto s = mat({-1, _}, {-2, _});
    CHECK(s.rows() == 1);
    CHECK(s.cols() == 2);
    // Single-arg index operator on reverse slice
    CHECK(s[0] == 15.0);
    CHECK(s[1] == 16.0);
    // Double-arg index operator on reverse slice
    CHECK(s(0, 0) == 15.0);
    CHECK(s(0, 1) == 16.0);
  }

  // Slice from start with step of 2
  {
    auto s = mat(_, {_, _, 2});
    CHECK(s.rows() == 4);
    CHECK(s.cols() == 2);
    CHECK(s ==
          Eigen::MatrixXd{{1.0, 3.0}, {5.0, 7.0}, {9.0, 11.0}, {13.0, 15.0}});
  }

  // Slice from end with negative step for row and column
  {
    auto s = mat({_, _, -1}, {_, _, -2});
    CHECK(s.rows() == 4);
    CHECK(s.cols() == 2);
    CHECK(s ==
          Eigen::MatrixXd{{16.0, 14.0}, {12.0, 10.0}, {8.0, 6.0}, {4.0, 2.0}});
  }

  // Slice from start and column -1
  {
    auto s = mat({1, _}, -1);
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 1);
    CHECK(s == Eigen::MatrixXd{{8.0}, {12.0}, {16.0}});
  }

  // Slice from start and column -2
  {
    auto s = mat({1, _}, -2);
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 1);
    CHECK(s == Eigen::MatrixXd{{7.0}, {11.0}, {15.0}});
  }
}

TEST_CASE("VariableMatrix - Subslicing", "[VariableMatrix]") {
  using namespace slp::slicing;

  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Block assignment
  CHECK(A({1, 3}, {1, 3}).rows() == 2);
  CHECK(A({1, 3}, {1, 3}).cols() == 2);
  A({1, 3}, {1, 3}) = Eigen::MatrixXd{{10.0, 11.0}, {12.0, 13.0}};

  Eigen::MatrixXd expected1{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 13.0}};
  CHECK(expected1 == A.value());

  // Block-of-block assignment
  CHECK(A({1, 3}, {1, 3})({1, _}, {1, _}).rows() == 1);
  CHECK(A({1, 3}, {1, 3})({1, _}, {1, _}).cols() == 1);
  A({1, 3}, {1, 3})({1, _}, {1, _}) = 14.0;

  Eigen::MatrixXd expected2{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 14.0}};
  CHECK(A.value() == expected2);
}

TEST_CASE("VariableMatrix - Iterators", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // VariableMatrix iterator
  CHECK(std::distance(A.begin(), A.end()) == 9);

  int i = 1;
  for (auto& elem : A) {
    CHECK(elem.value() == i);
    ++i;
  }

  // VariableMatrix const_iterator
  CHECK(std::distance(A.cbegin(), A.cend()) == 9);

  // Value() isn't const-qualified
#if 0
  i = 1;
  for (const auto& elem : A) {
    CHECK(elem.value() == i);
    ++i;
  }
#endif

  auto sub_A = A.block(2, 1, 1, 2);

  // VariableBlock iterator
  CHECK(std::distance(sub_A.begin(), sub_A.end()) == 2);

  i = 8;
  for (auto& elem : sub_A) {
    CHECK(elem.value() == i);
    ++i;
  }

  // VariableBlock const_iterator
  CHECK(std::distance(sub_A.begin(), sub_A.end()) == 2);

  // Value() isn't const-qualified
#if 0
  i = 8;
  for (const auto& elem : sub_A) {
    CHECK(elem.value() == i);
    ++i;
  }
#endif
}

TEST_CASE("VariableMatrix - Value", "[VariableMatrix]") {
  using namespace slp::slicing;

  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  Eigen::Matrix<double, 3, 3> expected{
      {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Full matrix
  CHECK(A.value() == expected);
  CHECK(A.value(3) == 4.0);
  CHECK(A.T().value(3) == 2.0);

  // Block
  CHECK(A.block(1, 1, 2, 2).value() == expected.block(1, 1, 2, 2));
  CHECK(A.block(1, 1, 2, 2).value(2) == 8.0);
  CHECK(A.T().block(1, 1, 2, 2).value(2) == 6.0);

  // Slice
  CHECK(A({1, 3}, {1, 3}).value() == expected.block(1, 1, 2, 2));
  CHECK(A({1, 3}, {1, 3}).value(2) == 8.0);
  CHECK(A({1, 3}, {1, 3}).T().value(2) == 6.0);

  // Block-of-block
  CHECK(A.block(1, 1, 2, 2).block(0, 1, 2, 1).value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A.block(1, 1, 2, 2).block(0, 1, 2, 1).value(1) == 9.0);
  CHECK(A.block(1, 1, 2, 2).T().block(0, 1, 2, 1).value(1) == 9.0);

  // Slice-of-slice
  CHECK(A({1, 3}, {1, 3})(_, {1, _}).value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A({1, 3}, {1, 3})(_, {1, _}).value(1) == 9.0);
  CHECK(A({1, 3}, {1, 3}).T()(_, {1, _}).value(1) == 9.0);
}

TEST_CASE("VariableMatrix - cwise_transform()", "[VariableMatrix]") {
  // VariableMatrix cwise_transform
  slp::VariableMatrix A{{-2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0}};

  slp::VariableMatrix result1 = A.cwise_transform(slp::abs);
  Eigen::Matrix<double, 2, 3> expected1{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};

  // Don't modify original matrix
  CHECK(A.value() == -expected1);

  CHECK(result1.value() == expected1);

  // VariableBlock cwise_transform
  auto sub_A = A.block(0, 0, 2, 2);

  slp::VariableMatrix result2 = sub_A.cwise_transform(slp::abs);
  Eigen::Matrix<double, 2, 2> expected2{{2.0, 3.0}, {5.0, 6.0}};

  // Don't modify original matrix
  CHECK(A.value() == -expected1);
  CHECK(sub_A.value() == -expected2);

  CHECK(result2.value() == expected2);
}

TEST_CASE("VariableMatrix - Zero() static function", "[VariableMatrix]") {
  auto A = slp::VariableMatrix::zero(2, 3);

  for (auto& elem : A) {
    CHECK(elem.value() == 0.0);
  }
}

TEST_CASE("VariableMatrix - Ones() static function", "[VariableMatrix]") {
  auto A = slp::VariableMatrix::ones(2, 3);

  for (auto& elem : A) {
    CHECK(elem.value() == 1.0);
  }
}

TEST_CASE("VariableMatrix - cwise_reduce()", "[VariableMatrix]") {
  slp::VariableMatrix A{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};
  slp::VariableMatrix B{{8.0, 9.0, 10.0}, {11.0, 12.0, 13.0}};
  slp::VariableMatrix result = slp::cwise_reduce(A, B, std::multiplies<>{});

  Eigen::Matrix<double, 2, 3> expected{{16.0, 27.0, 40.0}, {55.0, 72.0, 91.0}};
  CHECK(result.value() == expected);
}

TEST_CASE("VariableMatrix - Block() free function", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  slp::VariableMatrix B{{7.0}, {8.0}};

  slp::VariableMatrix mat1 = slp::block({{A, B}});
  Eigen::Matrix<double, 2, 4> expected1{{1.0, 2.0, 3.0, 7.0},
                                        {4.0, 5.0, 6.0, 8.0}};
  CHECK(mat1.rows() == 2);
  CHECK(mat1.cols() == 4);
  CHECK(mat1.value() == expected1);

  slp::VariableMatrix C{{9.0, 10.0, 11.0, 12.0}};

  slp::VariableMatrix mat2 = slp::block({{A, B}, {C}});
  Eigen::Matrix<double, 3, 4> expected2{
      {1.0, 2.0, 3.0, 7.0}, {4.0, 5.0, 6.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
  CHECK(mat2.rows() == 3);
  CHECK(mat2.cols() == 4);
  CHECK(mat2.value() == expected2);
}

template <int Rows>
void ExpectSolve(const Eigen::Matrix<double, Rows, Rows>& A,
                 const Eigen::Matrix<double, Rows, 1>& B) {
  INFO(std::format("Solve {}x{}", Rows, Rows));

  slp::VariableMatrix slp_A{A};
  slp::VariableMatrix slp_B{B};
  auto actual_X = slp::solve(slp_A, slp_B);

  Eigen::Matrix<double, Rows, 1> expected_X = A.householderQr().solve(B);

  CHECK(actual_X.rows() == expected_X.rows());
  CHECK(actual_X.cols() == expected_X.cols());
  CHECK((slp_A.value() * actual_X.value() - slp_B.value()).norm() < 1e-12);
}

TEST_CASE("VariableMatrix - Solve() free function", "[VariableMatrix]") {
  // 1x1 special case
  ExpectSolve(Eigen::Matrix<double, 1, 1>{{2.0}},
              Eigen::Matrix<double, 1, 1>{{5.0}});

  // 2x2 special case
  ExpectSolve(Eigen::Matrix<double, 2, 2>{{1.0, 2.0}, {3.0, 4.0}},
              Eigen::Matrix<double, 2, 1>{{5.0}, {6.0}});

  // 3x3 special case
  ExpectSolve(
      Eigen::Matrix<double, 3, 3>{
          {1.0, 2.0, 3.0}, {-4.0, -5.0, 6.0}, {7.0, 8.0, 9.0}},
      Eigen::Matrix<double, 3, 1>{{10.0}, {11.0}, {12.0}});

  // 4x4 special case
  ExpectSolve(Eigen::Matrix<double, 4, 4>{{1.0, 2.0, 3.0, -4.0},
                                          {-5.0, 6.0, 7.0, 8.0},
                                          {9.0, 10.0, 11.0, 12.0},
                                          {13.0, 14.0, 15.0, 16.0}},
              Eigen::Matrix<double, 4, 1>{{17.0}, {18.0}, {19.0}, {20.0}});

  // 5x5 general case
  ExpectSolve(
      Eigen::Matrix<double, 5, 5>{{1.0, 2.0, 3.0, -4.0, 5.0},
                                  {-5.0, 6.0, 7.0, 8.0, 9.0},
                                  {9.0, 10.0, 11.0, 12.0, 13.0},
                                  {13.0, 14.0, 15.0, 16.0, 17.0},
                                  {17.0, 18.0, 19.0, 20.0, 21.0}},
      Eigen::Matrix<double, 5, 1>{{21.0}, {22.0}, {23.0}, {24.0}, {25.0}});
}
