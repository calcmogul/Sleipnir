// Copyright (c) Sleipnir contributors

#include <format>
#include <functional>
#include <iterator>

#include <Eigen/Core>
#include <Eigen/QR>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>

TEST_CASE("VariableMatrix - Construct from Eigen::MatrixBase",
          "[VariableMatrix]") {
  sleipnir::VariableMatrix mat{
      Eigen::MatrixXd{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};

  Eigen::MatrixXd expected{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  CHECK(mat.Value() == expected);
}

TEST_CASE("VariableMatrix - Construct from Eigen::DiagonalBase",
          "[VariableMatrix]") {
  sleipnir::VariableMatrix mat{Eigen::VectorXd{{1.0, 2.0, 3.0}}.asDiagonal()};

  Eigen::MatrixXd expected{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
  CHECK(mat.Value() == expected);
}

TEST_CASE("VariableMatrix - Assignment to default", "[VariableMatrix]") {
  sleipnir::VariableMatrix mat;

  CHECK(mat.Rows() == 0);
  CHECK(mat.Cols() == 0);

  mat = sleipnir::VariableMatrix{2, 2};

  CHECK(mat.Rows() == 2);
  CHECK(mat.Cols() == 2);
  CHECK(mat[0, 0] == 0.0);
  CHECK(mat[0, 1] == 0.0);
  CHECK(mat[1, 0] == 0.0);
  CHECK(mat[1, 1] == 0.0);

  mat[0, 0] = 1.0;
  mat[0, 1] = 2.0;
  mat[1, 0] = 3.0;
  mat[1, 1] = 4.0;

  CHECK(mat[0, 0] == 1.0);
  CHECK(mat[0, 1] == 2.0);
  CHECK(mat[1, 0] == 3.0);
  CHECK(mat[1, 1] == 4.0);
}

TEST_CASE("VariableMatrix - Assignment aliasing", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0}, {3.0, 4.0}};
  sleipnir::VariableMatrix B{{5.0, 6.0}, {7.0, 8.0}};

  Eigen::MatrixXd expectedA{{1.0, 2.0}, {3.0, 4.0}};
  Eigen::MatrixXd expectedB{{5.0, 6.0}, {7.0, 8.0}};
  CHECK(A == expectedA);
  CHECK(B == expectedB);

  A = B;

  CHECK(A == expectedB);
  CHECK(B == expectedB);

  B[0, 0].SetValue(2.0);
  expectedB(0, 0) = 2.0;

  CHECK(A == expectedB);
  CHECK(B == expectedB);
}

TEST_CASE("VariableMatrix - Block() member function", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Block assignment
  A.Block(1, 1, 2, 2) = Eigen::Matrix<double, 2, 2>{{10.0, 11.0}, {12.0, 13.0}};

  Eigen::Matrix<double, 3, 3> expected1{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 13.0}};
  CHECK(A.Value() == expected1);

  // Block-of-block assignment
  A.Block(1, 1, 2, 2).Block(1, 1, 1, 1) = 14.0;

  Eigen::Matrix<double, 3, 3> expected2{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 14.0}};
  CHECK(A.Value() == expected2);
}

TEST_CASE("VariableMatrix - Slicing", "[VariableMatrix]") {
  using namespace sleipnir::slicing;

  sleipnir::VariableMatrix mat{
      {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  CHECK(mat.Rows() == 4);
  CHECK(mat.Cols() == 4);

  // Single-arg index operator on full matrix
  for (int i = 0; i < mat.Rows() * mat.Cols(); ++i) {
    CHECK(mat[i] == i + 1);
  }

  // Slice from start
  {
    auto s = mat[{1, _}, {2, _}];
    CHECK(s.Rows() == 3);
    CHECK(s.Cols() == 2);
    // Single-arg index operator on forward slice
    CHECK(s[0] == 7.0);
    CHECK(s[1] == 8.0);
    CHECK(s[2] == 11.0);
    CHECK(s[3] == 12.0);
    CHECK(s[4] == 15.0);
    CHECK(s[5] == 16.0);
    // Double-arg index operator on forward slice
    CHECK(s[0, 0] == 7.0);
    CHECK(s[0, 1] == 8.0);
    CHECK(s[1, 0] == 11.0);
    CHECK(s[1, 1] == 12.0);
    CHECK(s[2, 0] == 15.0);
    CHECK(s[2, 1] == 16.0);
  }

  // Slice from end
  {
    auto s = mat[{-1, _}, {-2, _}];
    CHECK(s.Rows() == 1);
    CHECK(s.Cols() == 2);
    // Single-arg index operator on reverse slice
    CHECK(s[0] == 15.0);
    CHECK(s[1] == 16.0);
    // Double-arg index operator on reverse slice
    CHECK(s[0, 0] == 15.0);
    CHECK(s[0, 1] == 16.0);
  }

  // Slice from start with step of 2
  {
    auto s = mat[{_}, {_, _, 2}];
    CHECK(s.Rows() == 4);
    CHECK(s.Cols() == 2);
    CHECK(s ==
          Eigen::MatrixXd{{1.0, 3.0}, {5.0, 7.0}, {9.0, 11.0}, {13.0, 15.0}});
  }

  // Slice from end with negative step for row and column
  {
    auto s = mat[{_, _, -1}, {_, _, -2}];
    CHECK(s.Rows() == 4);
    CHECK(s.Cols() == 2);
    CHECK(s ==
          Eigen::MatrixXd{{16.0, 14.0}, {12.0, 10.0}, {8.0, 6.0}, {4.0, 2.0}});
  }
}

TEST_CASE("VariableMatrix - Subslicing", "[VariableMatrix]") {
  using namespace sleipnir::slicing;

  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Block assignment
  CHECK(A[{1, 3}, {1, 3}].Rows() == 2);
  CHECK(A[{1, 3}, {1, 3}].Cols() == 2);
  A[{1, 3}, {1, 3}] = Eigen::MatrixXd{{10.0, 11.0}, {12.0, 13.0}};

  Eigen::MatrixXd expected1{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 13.0}};
  CHECK(expected1 == A.Value());

  // Block-of-block assignment
  CHECK(A[{1, 3}, {1, 3}][{1, _}, {1, _}].Rows() == 1);
  CHECK(A[{1, 3}, {1, 3}][{1, _}, {1, _}].Cols() == 1);
  A[{1, 3}, {1, 3}][{1, _}, {1, _}] = 14.0;

  Eigen::MatrixXd expected2{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 14.0}};
  CHECK(A.Value() == expected2);
}

TEST_CASE("VariableMatrix - Iterators", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // VariableMatrix iterator
  CHECK(std::distance(A.begin(), A.end()) == 9);

  int i = 1;
  for (auto& elem : A) {
    CHECK(elem.Value() == i);
    ++i;
  }

  // VariableMatrix const_iterator
  CHECK(std::distance(A.cbegin(), A.cend()) == 9);

  // Value() isn't const-qualified
#if 0
  i = 1;
  for (const auto& elem : A) {
    CHECK(elem.Value() == i);
    ++i;
  }
#endif

  auto Asub = A.Block(2, 1, 1, 2);

  // VariableBlock iterator
  CHECK(std::distance(Asub.begin(), Asub.end()) == 2);

  i = 8;
  for (auto& elem : Asub) {
    CHECK(elem.Value() == i);
    ++i;
  }

  // VariableBlock const_iterator
  CHECK(std::distance(Asub.begin(), Asub.end()) == 2);

  // Value() isn't const-qualified
#if 0
  i = 8;
  for (const auto& elem : Asub) {
    CHECK(elem.Value() == i);
    ++i;
  }
#endif
}

TEST_CASE("VariableMatrix - Value", "[VariableMatrix]") {
  using namespace sleipnir::slicing;

  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  Eigen::Matrix<double, 3, 3> expected{
      {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Full matrix
  CHECK(A.Value() == expected);
  CHECK(A.Value(3) == 4.0);
  CHECK(A.T().Value(3) == 2.0);

  // Block
  CHECK(A.Block(1, 1, 2, 2).Value() == expected.block(1, 1, 2, 2));
  CHECK(A.Block(1, 1, 2, 2).Value(2) == 8.0);
  CHECK(A.T().Block(1, 1, 2, 2).Value(2) == 6.0);

  // Slice
  CHECK(A[{1, 3}, {1, 3}].Value() == expected.block(1, 1, 2, 2));
  CHECK(A[{1, 3}, {1, 3}].Value(2) == 8.0);
  CHECK(A[{1, 3}, {1, 3}].T().Value(2) == 6.0);

  // Block-of-block
  CHECK(A.Block(1, 1, 2, 2).Block(0, 1, 2, 1).Value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A.Block(1, 1, 2, 2).Block(0, 1, 2, 1).Value(1) == 9.0);
  CHECK(A.Block(1, 1, 2, 2).T().Block(0, 1, 2, 1).Value(1) == 9.0);

  // Slice-of-slice
  CHECK(A[{1, 3}, {1, 3}][{_}, {1, _}].Value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A[{1, 3}, {1, 3}][{_}, {1, _}].Value(1) == 9.0);
  CHECK(A[{1, 3}, {1, 3}].T()[{_}, {1, _}].Value(1) == 9.0);
}

TEST_CASE("VariableMatrix - CwiseTransform()", "[VariableMatrix]") {
  // VariableMatrix CwiseTransform
  sleipnir::VariableMatrix A{{-2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0}};

  sleipnir::VariableMatrix result1 = A.CwiseTransform(sleipnir::abs);
  Eigen::Matrix<double, 2, 3> expected1{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};

  // Don't modify original matrix
  CHECK(A.Value() == -expected1);

  CHECK(result1.Value() == expected1);

  // VariableBlock CwiseTransform
  auto Asub = A.Block(0, 0, 2, 2);

  sleipnir::VariableMatrix result2 = Asub.CwiseTransform(sleipnir::abs);
  Eigen::Matrix<double, 2, 2> expected2{{2.0, 3.0}, {5.0, 6.0}};

  // Don't modify original matrix
  CHECK(A.Value() == -expected1);
  CHECK(Asub.Value() == -expected2);

  CHECK(result2.Value() == expected2);
}

TEST_CASE("VariableMatrix - Zero() static function", "[VariableMatrix]") {
  auto A = sleipnir::VariableMatrix::Zero(2, 3);

  for (auto& elem : A) {
    CHECK(elem.Value() == 0.0);
  }
}

TEST_CASE("VariableMatrix - Ones() static function", "[VariableMatrix]") {
  auto A = sleipnir::VariableMatrix::Ones(2, 3);

  for (auto& elem : A) {
    CHECK(elem.Value() == 1.0);
  }
}

TEST_CASE("VariableMatrix - CwiseReduce()", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};
  sleipnir::VariableMatrix B{{8.0, 9.0, 10.0}, {11.0, 12.0, 13.0}};
  sleipnir::VariableMatrix result =
      sleipnir::CwiseReduce(A, B, std::multiplies<>{});

  Eigen::Matrix<double, 2, 3> expected{{16.0, 27.0, 40.0}, {55.0, 72.0, 91.0}};
  CHECK(result.Value() == expected);
}

TEST_CASE("VariableMatrix - Block() free function", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  sleipnir::VariableMatrix B{{7.0}, {8.0}};

  sleipnir::VariableMatrix mat1 = sleipnir::Block({{A, B}});
  Eigen::Matrix<double, 2, 4> expected1{{1.0, 2.0, 3.0, 7.0},
                                        {4.0, 5.0, 6.0, 8.0}};
  CHECK(mat1.Rows() == 2);
  CHECK(mat1.Cols() == 4);
  CHECK(mat1.Value() == expected1);

  sleipnir::VariableMatrix C{{9.0, 10.0, 11.0, 12.0}};

  sleipnir::VariableMatrix mat2 = sleipnir::Block({{A, B}, {C}});
  Eigen::Matrix<double, 3, 4> expected2{
      {1.0, 2.0, 3.0, 7.0}, {4.0, 5.0, 6.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
  CHECK(mat2.Rows() == 3);
  CHECK(mat2.Cols() == 4);
  CHECK(mat2.Value() == expected2);
}

template <int Rows>
void ExpectSolve(const Eigen::Matrix<double, Rows, Rows>& A,
                 const Eigen::Matrix<double, Rows, 1>& B) {
  INFO(std::format("Solve {}x{}", Rows, Rows));

  sleipnir::VariableMatrix slpA{A};
  sleipnir::VariableMatrix slpB{B};
  auto actualX = sleipnir::Solve(slpA, slpB);

  Eigen::Matrix<double, Rows, 1> expectedX = A.householderQr().solve(B);

  CHECK(actualX.Rows() == expectedX.rows());
  CHECK(actualX.Cols() == expectedX.cols());
  CHECK((slpA.Value() * actualX.Value() - slpB.Value()).norm() < 1e-12);
  CHECK((actualX.Value() - expectedX).norm() < 1e-12);
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
