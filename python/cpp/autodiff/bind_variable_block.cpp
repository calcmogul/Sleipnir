// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>

#include <fmt/format.h>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

void bind_variable_block(
    nb::class_<VariableBlock<VariableMatrix<double>>>& cls) {
  using namespace nb::literals;
  using MatrixXi64 = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixXi32 = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>;

  cls.def(
      "set_value",
      [](VariableBlock<VariableMatrix<double>>& self, double value) {
        self.set_value(value);
      },
      "value"_a, DOC(slp, VariableBlock, set_value));
  cls.def(
      "set_value",
      [](VariableBlock<VariableMatrix<double>>& self,
         nb::DRef<Eigen::MatrixXd> values) { self.set_value(values); },
      "values"_a, DOC(slp, VariableBlock, set_value, 2));
  for_each_type<Eigen::MatrixXf, MatrixXi64, MatrixXi32>([&]<typename T> {
    cls.def(
        "set_value",
        [](VariableBlock<VariableMatrix<double>>& self, nb::DRef<T> values) {
          self.set_value(values.template cast<double>());
        },
        "values"_a, DOC(slp, VariableBlock, set_value, 2));
  });
  cls.def(
      "__setitem__",
      [](VariableBlock<VariableMatrix<double>>& self, int row,
         const Variable<double>& value) {
        if (row < 0) {
          row += self.size();
        }
        return self[row] = value;
      },
      "row"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](VariableBlock<VariableMatrix<double>>& self, nb::tuple slices,
         nb::object value) {
        if (slices.size() != 2) {
          throw nb::index_error(
              fmt::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        Slice row_slice;
        int row_slice_length;
        Slice col_slice;
        int col_slice_length;

        // Row slice
        const auto& row_elem = slices[0];
        if (auto py_row_slice = try_cast<nb::slice>(row_elem)) {
          auto t = py_row_slice.value().compute(self.rows());
          row_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          row_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(row_elem);
          if (start < 0) {
            start += self.rows();
          }
          row_slice = Slice{start, start + 1};
          row_slice_length = 1;
        }

        // Column slice
        const auto& col_elem = slices[1];
        if (auto py_col_slice = try_cast<nb::slice>(col_elem)) {
          auto t = py_col_slice.value().compute(self.cols());
          col_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          col_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(col_elem);
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
          throw nb::value_error(
              "VariableBlock.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.def(
      "__getitem__",
      [](VariableBlock<VariableMatrix<double>>& self,
         int row) -> Variable<double>& {
        if (row < 0) {
          row += self.size();
        }
        return self[row];
      },
      nb::keep_alive<0, 1>(), "row"_a,
      DOC(slp, VariableBlock, operator, array, 3));
  cls.def(
      "__getitem__",
      [](VariableBlock<VariableMatrix<double>>& self,
         nb::tuple slices) -> nb::object {
        if (slices.size() != 2) {
          throw nb::index_error(
              fmt::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (nb::isinstance<int>(slices[0]) && nb::isinstance<int>(slices[1])) {
          int row = nb::cast<int>(slices[0]);
          int col = nb::cast<int>(slices[1]);

          if (row >= self.rows() || col >= self.cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row += self.rows();
          }
          if (col < 0) {
            col += self.cols();
          }
          return nb::cast(self[row, col]);
        }

        Slice row_slice;
        int row_slice_length;
        Slice col_slice;
        int col_slice_length;

        // Row slice
        const auto& row_elem = slices[0];
        if (auto py_row_slice = try_cast<nb::slice>(row_elem)) {
          auto t = py_row_slice.value().compute(self.rows());
          row_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          row_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(row_elem);
          if (start < 0) {
            start += self.rows();
          }
          row_slice = Slice{start, start + 1};
          row_slice_length = 1;
        }

        // Column slice
        const auto& col_elem = slices[1];
        if (auto py_col_slice = try_cast<nb::slice>(col_elem)) {
          auto t = py_col_slice.value().compute(self.cols());
          col_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          col_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(col_elem);
          if (start < 0) {
            start += self.cols();
          }
          col_slice = Slice{start, start + 1};
          col_slice_length = 1;
        }

        return nb::cast(
            self[row_slice, row_slice_length, col_slice, col_slice_length]);
      },
      nb::keep_alive<0, 1>(), DOC(slp, VariableBlock, operator, array),
      "slices"_a);
  cls.def("row",
          nb::overload_cast<int>(&VariableBlock<VariableMatrix<double>>::row),
          "row"_a, DOC(slp, VariableBlock, row));
  cls.def("col",
          nb::overload_cast<int>(&VariableBlock<VariableMatrix<double>>::col),
          "col"_a, DOC(slp, VariableBlock, col));

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](VariableBlock<VariableMatrix<double>>& self, nb::object ufunc,
         nb::str method, nb::args inputs, const nb::kwargs&) -> nb::object {
        std::string method_name = nb::cast<std::string>(method);
        std::string ufunc_name =
            nb::cast<std::string>(ufunc.attr("__repr__")());

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
            nb::cast<std::string>(inputs[0].attr("__repr__")());
        std::string input2_name =
            nb::cast<std::string>(inputs[1].attr("__repr__")());
        throw nb::value_error(
            fmt::format("VariableBlock: numpy method {}, ufunc {} not "
                        "implemented for ({}, {})",
                        method_name, ufunc_name, input1_name, input2_name)
                .c_str());
      },
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(-nb::self);

  // Matrix-scalar/scalar-matrix operations
  for_each_type<double, int, Variable<double>>([&]<typename T> {
    cls.def(nb::self * T(), "rhs"_a);
    cls.def(T() * nb::self, "lhs"_a);
    cls.def(nb::self / T(), "rhs"_a);
    cls.def(nb::self /= T(), "rhs"_a, DOC(slp, VariableBlock, operator, idiv));
  });
  for_each_type<double, int, const Variable<double>&,
                nb::DRef<Eigen::MatrixXd>>([&]<typename T> {
    cls.def(
        "__add__",
        [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs + Variable<double>{rhs};
          } else {
            return lhs + rhs;
          }
        },
        nb::is_operator(), "rhs"_a);
    cls.def(
        "__radd__",
        [](const VariableBlock<VariableMatrix<double>>& rhs, T&& lhs) {
          if constexpr (ScalarLike<T>) {
            return Variable<double>{lhs} + rhs;
          } else {
            return lhs + rhs;
          }
        },
        nb::is_operator(), "lhs"_a);
    cls.def(
        "__sub__",
        [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs - Variable<double>{rhs};
          } else {
            return lhs - rhs;
          }
        },
        nb::is_operator(), "rhs"_a);
    cls.def(
        "__rsub__",
        [](const VariableBlock<VariableMatrix<double>>& rhs, T&& lhs) {
          if constexpr (ScalarLike<T>) {
            return Variable<double>{lhs} - rhs;
          } else {
            return lhs - rhs;
          }
        },
        nb::is_operator(), "lhs"_a);
  });

  cls.def(
      "__pow__",
      [](const VariableBlock<VariableMatrix<double>>& self, int power) {
        return self.cwise_transform(
            [=](const auto& elem) { return pow(elem, power); });
      },
      nb::is_operator(), "power"_a);
  cls.def_prop_ro("T", &VariableBlock<VariableMatrix<double>>::T,
                  DOC(slp, VariableBlock, T));
  cls.def("rows", &VariableBlock<VariableMatrix<double>>::rows,
          DOC(slp, VariableBlock, rows));
  cls.def("cols", &VariableBlock<VariableMatrix<double>>::cols,
          DOC(slp, VariableBlock, cols));
  cls.def_prop_ro("shape",
                  [](const VariableBlock<VariableMatrix<double>>& self) {
                    return nb::make_tuple(self.rows(), self.cols());
                  });
  cls.def("value",
          nb::overload_cast<int, int>(
              &VariableBlock<VariableMatrix<double>>::value),
          "row"_a, "col"_a, DOC(slp, VariableBlock, value));
  cls.def("value",
          nb::overload_cast<int>(&VariableBlock<VariableMatrix<double>>::value),
          "index"_a, DOC(slp, VariableBlock, value, 2));
  cls.def("value",
          nb::overload_cast<>(&VariableBlock<VariableMatrix<double>>::value),
          DOC(slp, VariableBlock, value, 3));
  cls.def(
      "cwise_map",
      [](const VariableBlock<VariableMatrix<double>>& self,
         const std::function<Variable<double>(const Variable<double>& x)>&
             unary_op) { return self.cwise_transform(unary_op); },
      "func"_a, DOC(slp, VariableBlock, cwise_transform));

  // Comparison operators
  for_each_type<nb::detail::self_t, double, int, Variable<double>>(
      [&]<typename T> {
        cls.def(nb::self == T(), "rhs"_a, DOC(slp, operator, eq));
        cls.def(nb::self < T(), "rhs"_a, DOC(slp, operator, lt));
        cls.def(nb::self <= T(), "rhs"_a, DOC(slp, operator, le));
        cls.def(nb::self > T(), "rhs"_a, DOC(slp, operator, gt));
        cls.def(nb::self >= T(), "rhs"_a, DOC(slp, operator, ge));
        if constexpr (!std::same_as<nb::detail::self_t, T>) {
          cls.def(T() == nb::self, "rhs"_a, DOC(slp, operator, eq));
          cls.def(T() < nb::self, "rhs"_a, DOC(slp, operator, lt));
          cls.def(T() <= nb::self, "rhs"_a, DOC(slp, operator, le));
          cls.def(T() > nb::self, "rhs"_a, DOC(slp, operator, gt));
          cls.def(T() >= nb::self, "rhs"_a, DOC(slp, operator, ge));
        }
      });
  for_each_type<const VariableMatrix<double>&, nb::DRef<Eigen::MatrixXd>>(
      [&]<typename T> {
        cls.def(
            "__eq__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs == rhs;
            },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, eq));
        cls.def(
            "__lt__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs < rhs;
            },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, lt));
        cls.def(
            "__le__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs <= rhs;
            },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, le));
        cls.def(
            "__gt__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs > rhs;
            },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, gt));
        cls.def(
            "__ge__",
            [](const VariableBlock<VariableMatrix<double>>& lhs, T&& rhs) {
              return lhs >= rhs;
            },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, ge));
      });

  cls.def("__len__", &VariableBlock<VariableMatrix<double>>::rows,
          DOC(slp, VariableBlock, rows));

  cls.def(
      "__iter__",
      [](const VariableBlock<VariableMatrix<double>>& self) {
        return nb::make_iterator(
            nb::type<VariableBlock<VariableMatrix<double>>>(), "value_iterator",
            self.begin(), self.end());
      },
      nb::keep_alive<0, 1>());
}

}  // namespace slp
