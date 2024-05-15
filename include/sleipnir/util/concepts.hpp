// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <type_traits>

#include <Eigen/Core>

namespace sleipnir {

template <typename T>
concept ScalarLike = requires(std::decay_t<T> t) {
  t + 1.0;
  t = 1.0;
};

template <typename T>
concept EigenMatrixLike =
    std::derived_from<std::decay_t<T>, Eigen::MatrixBase<std::decay_t<T>>>;

template <typename T>
concept SleipnirMatrixLike = requires(T t, int rows, int cols) {
  t.rows();
  t.cols();
  t[rows, cols];
} && !EigenMatrixLike<T>;

template <typename T>
concept MatrixLike = SleipnirMatrixLike<T> || EigenMatrixLike<T>;

template <typename T>
concept EigenSolver = requires(T t) { t.solve(Eigen::VectorXd{}); };

}  // namespace sleipnir
