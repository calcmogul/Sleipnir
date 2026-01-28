// Copyright (c) Sleipnir contributors

#include <format>
#include <string>

#include <emscripten/bind.h>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "for_each_type.hpp"
#include "try_cast.hpp"

namespace em = emscripten;

namespace slp {

void bind_variable_block(
    em::class_<VariableBlock<VariableMatrix<double>>>& cls) {
  using namespace em::literals;
  using MatrixXi64 = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixXi32 = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>;

  cls.function(
      "set_value",
      [](VariableBlock<VariableMatrix<double>>& self, double value) {
        self.set_value(value);
      },
      "value"_a);
  cls.function(
      "set_value",
      [](VariableBlock<VariableMatrix<double>>& self,
         em::DRef<Eigen::MatrixXd> values) { self.set_value(values); },
      "values"_a);
  for_each_type<Eigen::MatrixXf, MatrixXi64, MatrixXi32>([&]<typename T> {
    cls.function(
        "set_value",
        [](VariableBlock<VariableMatrix<double>>& self, em::DRef<T> values) {
          self.set_value(values.template cast<double>());
        },
        "values"_a);
  });
  cls.function(
      "__setitem__",
      [](VariableBlock<VariableMatrix<double>>& self, int row,
         const Variable<double>& value) {
        if (row < 0) {
          row += self.size();
        }
        return self[row] = value;
      },
      "row"_a, "value"_a);
  cls.function(
      "__setitem__",
      [](VariableBlock<VariableMatrix<double>>& self, em::tuple slices,
         em::object value) {
        if (slices.size() != 2) {
          throw em::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        Slice row_slice;
        int row_slice_length;
        Slice col_slice;
        int col_slice_length;

        // Row slice
        const auto& row_elem = slices[0];
        if (auto py_row_slice = try_cast<em::slice>(row_elem)) {
          auto t = py_row_slice.value().compute(self.rows());
          row_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          row_slice_length = t.get<3>();
        } else {
          int start = em::cast<int>(row_elem);
          if (start < 0) {
            start += self.rows();
          }
          row_slice = Slice{start, start + 1};
          row_slice_length = 1;
        }

        // Column slice
        const auto& col_elem = slices[1];
        if (auto py_col_slice = try_cast<em::slice>(col_elem)) {
          auto t = py_col_slice.value().compute(self.cols());
          col_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          col_slice_length = t.get<3>();
        } else {
          int start = em::cast<int>(col_elem);
          if (start < 0) {
            start += self.cols();
          }
          col_slice = Slice{start, start + 1};
          col_slice_length = 1;
        }

        auto lhs =
            self[row_slice, row_slice_length, col_slice, col_slice_length];
        if (auto rhs = try_cast<VariableMatrix<double>>(value)) {
          lhs = rhs.value();
        } else if (auto rhs =
                       try_cast<VariableBlock<VariableMatrix<double>>>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<double>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<float>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int64_t>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int32_t>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast<Variable<double>>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast<double>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast<int>(value)) {
          lhs = rhs.value();
        } else {
          throw em::value_error(
              "VariableBlock.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.function(
      "__getitem__",
      [](VariableBlock<VariableMatrix<double>>& self,
         int row) -> Variable<double>& {
        if (row < 0) {
          row += self.size();
        }
        return self[row];
      },
      em::keep_alive<0, 1>(), "row"_a);
  cls.function(
      "__getitem__",
      [](VariableBlock<VariableMatrix<double>>& self,
         em::tuple slices) -> em::object {
        if (slices.size() != 2) {
          throw em::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (em::isinstance<int>(slices[0]) && em::isinstance<int>(slices[1])) {
          int row = em::cast<int>(slices[0]);
          int col = em::cast<int>(slices[1]);

          if (row >= self.rows() || col >= self.cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row += self.rows();
          }
          if (col < 0) {
            col += self.cols();
          }
          return em::cast(self[row, col]);
        }

        Slice row_slice;
        int row_slice_length;
        Slice col_slice;
        int col_slice_length;

        // Row slice
        const auto& row_elem = slices[0];
        if (auto py_row_slice = try_cast<em::slice>(row_elem)) {
          auto t = py_row_slice.value().compute(self.rows());
          row_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          row_slice_length = t.get<3>();
        } else {
          int start = em::cast<int>(row_elem);
          if (start < 0) {
            start += self.rows();
          }
          row_slice = Slice{start, start + 1};
          row_slice_length = 1;
        }

        // Column slice
        const auto& col_elem = slices[1];
        if (auto py_col_slice = try_cast<em::slice>(col_elem)) {
          auto t = py_col_slice.value().compute(self.cols());
          col_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          col_slice_length = t.get<3>();
        } else {
          int start = em::cast<int>(col_elem);
          if (start < 0) {
            start += self.cols();
          }
          col_slice = Slice{start, start + 1};
          col_slice_length = 1;
        }

        return em::cast(
            self[row_slice, row_slice_length, col_slice, col_slice_length]);
      },
      em::keep_alive<0, 1>(), "slices"_a);
  cls.function(
      "row",
      em::overload_cast<int>(&VariableBlock<VariableMatrix<double>>::row),
      "row"_a);
  cls.function(
      "col",
      em::overload_cast<int>(&VariableBlock<VariableMatrix<double>>::col),
      "col"_a);

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.function(
      "__array_ufunc__",
      [](VariableBlock<VariableMatrix<double>>& self, em::object ufunc,
         em::str method, em::args inputs, const em::kwargs&) -> em::object {
        std::string method_name = em::cast<std::string>(method);
        std::string ufunc_name =
            em::cast<std::string>(ufunc.attr("__repr__")());

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs * self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self * rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs + self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self + rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs - self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self - rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs == self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self == rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs < self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self < rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs <= self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self <= rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs > self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self > rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs >= self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self >= rhs; }, inputs[1])) {
              return result.value();
            }
          }
        }

        std::string input1_name =
            em::cast<std::string>(inputs[0].attr("__repr__")());
        std::string input2_name =
            em::cast<std::string>(inputs[1].attr("__repr__")());
        throw em::value_error(
            std::format("VariableBlock: numpy method {}, ufunc {} not "
                        "implemented for ({}, {})",
                        method_name, ufunc_name, input1_name, input2_name)
                .c_str());
      },
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  cls.function(em::self + em::self, "rhs"_a);
  cls.function(em::self - em::self, "rhs"_a);
  cls.function(-em::self);

  // Matrix-scalar/scalar-matrix operations
  for_each_type<double, int, Variable<double>>([&]<typename T> {
    cls.function(em::self * T(), "rhs"_a);
    cls.function(T() * em::self, "lhs"_a);
    cls.function(em::self / T(), "rhs"_a);
    cls.function(em::self /= T(), "rhs"_a);
  });
  for_each_type<double, int, const Variable<double>&,
                em::DRef<Eigen::MatrixXd>>([&]<typename T> {
    cls.function(
        "__add__",
        [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs + Variable<double>{rhs};
          } else {
            return lhs + rhs;
          }
        },
        em::is_operator(), "rhs"_a);
    cls.function(
        "__radd__",
        [](const VariableBlock<VariableMatrix<double>>& rhs, T&& lhs) {
          if constexpr (ScalarLike<T>) {
            return Variable<double>{lhs} + rhs;
          } else {
            return lhs + rhs;
          }
        },
        em::is_operator(), "lhs"_a);
    cls.function(
        "__sub__",
        [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs - Variable<double>{rhs};
          } else {
            return lhs - rhs;
          }
        },
        em::is_operator(), "rhs"_a);
    cls.function(
        "__rsub__",
        [](const VariableBlock<VariableMatrix<double>>& rhs, T&& lhs) {
          if constexpr (ScalarLike<T>) {
            return Variable<double>{lhs} - rhs;
          } else {
            return lhs - rhs;
          }
        },
        em::is_operator(), "lhs"_a);
  });

  cls.function(
      "__pow__",
      [](const VariableBlock<VariableMatrix<double>>& self, int power) {
        return self.cwise_transform(
            [=](const auto& elem) { return pow(elem, power); });
      },
      em::is_operator(), "power"_a);
  cls.property("T", &VariableBlock<VariableMatrix<double>>::T);
  cls.function("rows", &VariableBlock<VariableMatrix<double>>::rows);
  cls.function("cols", &VariableBlock<VariableMatrix<double>>::cols);
  cls.property("shape", [](const VariableBlock<VariableMatrix<double>>& self) {
    return em::make_tuple(self.rows(), self.cols());
  });
  cls.function("value",
               em::overload_cast<int, int>(
                   &VariableBlock<VariableMatrix<double>>::value),
               "row"_a, "col"_a);
  cls.function(
      "value",
      em::overload_cast<int>(&VariableBlock<VariableMatrix<double>>::value),
      "index"_a);
  cls.function("value", em::overload_cast<>(
                            &VariableBlock<VariableMatrix<double>>::value));
  cls.function(
      "cwise_map",
      [](const VariableBlock<VariableMatrix<double>>& self,
         const std::function<Variable<double>(const Variable<double>& x)>&
             unary_op) { return self.cwise_transform(unary_op); },
      "func"_a);

  // Comparison operators
  for_each_type<em::detail::self_t, double, int, Variable<double>>(
      [&]<typename T> {
        cls.function(em::self == T(), "rhs"_a);
        cls.function(em::self < T(), "rhs"_a);
        cls.function(em::self <= T(), "rhs"_a);
        cls.function(em::self > T(), "rhs"_a);
        cls.function(em::self >= T(), "rhs"_a);
        if constexpr (!std::same_as<em::detail::self_t, T>) {
          cls.function(T() == em::self, "rhs"_a);
          cls.function(T() < em::self, "rhs"_a);
          cls.function(T() <= em::self, "rhs"_a);
          cls.function(T() > em::self, "rhs"_a);
          cls.function(T() >= em::self, "rhs"_a);
        }
      });
  for_each_type<const VariableMatrix<double>&, em::DRef<Eigen::MatrixXd>>(
      [&]<typename T> {
        cls.function(
            "__eq__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs == rhs;
            },
            em::is_operator(), "rhs"_a);
        cls.function(
            "__lt__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs < rhs;
            },
            em::is_operator(), "rhs"_a);
        cls.function(
            "__le__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs <= rhs;
            },
            em::is_operator(), "rhs"_a);
        cls.function(
            "__gt__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs > rhs;
            },
            em::is_operator(), "rhs"_a);
        cls.function(
            "__ge__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs >= rhs;
            },
            em::is_operator(), "rhs"_a);
      });

  cls.function("__len__", &VariableBlock<VariableMatrix<double>>::rows);

  cls.function(
      "__iter__",
      [](const VariableBlock<VariableMatrix<double>>& self) {
        return em::make_iterator(
            em::type<VariableBlock<VariableMatrix<double>>>(), "value_iterator",
            self.begin(), self.end());
      },
      em::keep_alive<0, 1>());
}

}  // namespace slp
