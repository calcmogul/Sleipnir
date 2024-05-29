// Copyright (c) Sleipnir contributors

#include <string>
#include <vector>

#include <fmt/format.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "docstrings.hpp"
#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

void bind_variable_matrix(nb::module_& autodiff,
                          nb::class_<VariableMatrix<double>>& cls) {
  using namespace nb::literals;
  using MatrixXi64 = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixXi32 = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>;

  cls.def(nb::init<>(), DOC(slp, VariableMatrix, VariableMatrix));
  cls.def(nb::init<int>(), "rows"_a,
          DOC(slp, VariableMatrix, VariableMatrix, 2));
  cls.def(nb::init<int, int>(), "rows"_a, "cols"_a,
          DOC(slp, VariableMatrix, VariableMatrix, 3));
  cls.def(nb::init<const std::vector<std::vector<Variable<double>>>&>(),
          "list"_a, DOC(slp, VariableMatrix, VariableMatrix, 5));
  cls.def(nb::init<const std::vector<std::vector<double>>&>(), "list"_a,
          DOC(slp, VariableMatrix, VariableMatrix, 6));
  cls.def(nb::init<nb::DRef<Eigen::MatrixXd>>(), "values"_a,
          DOC(slp, VariableMatrix, VariableMatrix, 8));
  cls.def(nb::init<nb::DRef<Eigen::MatrixXf>>(), "values"_a,
          DOC(slp, VariableMatrix, VariableMatrix, 8));
  cls.def(nb::init<const Variable<double>&>(), "variable"_a,
          DOC(slp, VariableMatrix, VariableMatrix, 10));
  cls.def(nb::init<const VariableBlock<VariableMatrix<double>>&>(), "values"_a,
          DOC(slp, VariableMatrix, VariableMatrix, 12));
  cls.def(
      "set_value",
      [](VariableMatrix<double>& self, nb::DRef<Eigen::MatrixXd> values) {
        self.set_value(values);
      },
      "values"_a, DOC(slp, VariableMatrix, set_value));
  cls.def(
      "set_value",
      [](VariableMatrix<double>& self, nb::DRef<Eigen::MatrixXf> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(slp, VariableMatrix, set_value));
  cls.def(
      "set_value",
      [](VariableMatrix<double>& self, nb::DRef<MatrixXi64> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(slp, VariableMatrix, set_value));
  cls.def(
      "set_value",
      [](VariableMatrix<double>& self, nb::DRef<MatrixXi32> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(slp, VariableMatrix, set_value));
  cls.def(
      "__setitem__",
      [](VariableMatrix<double>& self, int row, const Variable<double>& value) {
        if (row < 0) {
          row += self.size();
        }
        return self[row] = value;
      },
      "row"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](VariableMatrix<double>& self, nb::tuple slices, nb::object value) {
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

        if (auto rhs = try_cast<VariableMatrix<double>>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs =
                       try_cast<VariableBlock<VariableMatrix<double>>>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<double>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<float>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int64_t>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int32_t>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs = try_cast<Variable<double>>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs = try_cast<double>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else if (auto rhs = try_cast<int>(value)) {
          self[row_slice, row_slice_length, col_slice, col_slice_length] =
              rhs.value();
        } else {
          throw nb::value_error(
              "VariableMatrix.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.def(
      "__getitem__",
      [](VariableMatrix<double>& self, int row) -> Variable<double>& {
        if (row < 0) {
          row += self.size();
        }
        return self[row];
      },
      nb::keep_alive<0, 1>(), "row"_a,
      DOC(slp, VariableMatrix, operator, array, 3));
  cls.def(
      "__getitem__",
      [](VariableMatrix<double>& self, nb::tuple slices) -> nb::object {
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
      nb::keep_alive<0, 1>(), "slices"_a,
      DOC(slp, VariableMatrix, operator, array));
  cls.def("row", nb::overload_cast<int>(&VariableMatrix<double>::row), "row"_a,
          DOC(slp, VariableMatrix, row));
  cls.def("col", nb::overload_cast<int>(&VariableMatrix<double>::col), "col"_a,
          DOC(slp, VariableMatrix, col));

  // Matrix-matrix multiplication
  cls.def(
      "__mul__",
      [](const VariableMatrix<double>& lhs, const VariableMatrix<double>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__mul__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__mul__",
      [](const VariableBlock<VariableMatrix<double>>& lhs,
         const VariableMatrix<double>& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__mul__",
      [](const VariableBlock<VariableMatrix<double>>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableMatrix<double>& rhs, const VariableMatrix<double>& lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableMatrix<double>& rhs,
         const VariableBlock<VariableMatrix<double>>& lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableBlock<VariableMatrix<double>>& rhs,
         const VariableMatrix<double>& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableBlock<VariableMatrix<double>>& rhs,
         const VariableBlock<VariableMatrix<double>>& lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableMatrix<double>& lhs, const VariableMatrix<double>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableBlock<VariableMatrix<double>>& lhs,
         const VariableMatrix<double>& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableBlock<VariableMatrix<double>>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](VariableMatrix<double>& self, nb::object ufunc, nb::str method,
         nb::args inputs, const nb::kwargs&) -> nb::object {
        std::string method_name = nb::cast<std::string>(method);
        std::string ufunc_name =
            nb::cast<std::string>(ufunc.attr("__repr__")());

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            }
          }
        }

        std::string input1_name =
            nb::cast<std::string>(inputs[0].attr("__repr__")());
        std::string input2_name =
            nb::cast<std::string>(inputs[1].attr("__repr__")());
        throw nb::value_error(
            fmt::format("VariableMatrix: numpy method {}, ufunc {} not "
                        "implemented for ({}, {})",
                        method_name, ufunc_name, input1_name, input2_name)
                .c_str());
        return nb::cast(VariableMatrix{self});
      },
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  // Matrix-scalar multiplication
  cls.def(nb::self * Variable<double>(), "rhs"_a);
  cls.def(nb::self * double(), "rhs"_a);

  // Scalar-matrix multiplication
  cls.def(Variable<double>() * nb::self, "lhs"_a);
  cls.def(double() * nb::self, "lhs"_a);

  // Matrix-scalar division
  cls.def(nb::self / Variable<double>(), "rhs"_a);
  cls.def(nb::self / double(), "rhs"_a);
  cls.def(nb::self /= Variable<double>(), "rhs"_a,
          DOC(slp, VariableMatrix, operator, idiv));
  cls.def(nb::self /= double(), "rhs"_a,
          DOC(slp, VariableMatrix, operator, idiv));

  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix<double>& lhs, const Variable<double>& rhs) {
        return lhs + VariableMatrix{rhs};
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix<double>& rhs, const Variable<double>& lhs) {
        return VariableMatrix{lhs} + rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__add__",
      [](double lhs, const VariableMatrix<double>& rhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix<double>& rhs, double lhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs)
          -> VariableMatrix<double> { return lhs + rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix<double>& rhs, nb::DRef<Eigen::MatrixXd> lhs)
          -> VariableMatrix<double> { return lhs + rhs; },
      nb::is_operator(), "lhs"_a);

  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix<double>& lhs, const Variable<double>& rhs) {
        return lhs - VariableMatrix{rhs};
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rsub__",
      [](const VariableMatrix<double>& rhs, const Variable<double>& lhs) {
        return VariableMatrix{lhs} - rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs)
          -> VariableMatrix<double> { return lhs - rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rsub__",
      [](const VariableMatrix<double>& rhs, nb::DRef<Eigen::MatrixXd> lhs)
          -> VariableMatrix<double> { return lhs - rhs; },
      nb::is_operator(), "lhs"_a);

  cls.def(-nb::self);
  cls.def(
      "__pow__",
      [](const VariableMatrix<double>& self, int power) {
        return self.cwise_transform(
            [=](const auto& elem) { return pow(elem, power); });
      },
      nb::is_operator(), "power"_a);
  cls.def_prop_ro("T", &VariableMatrix<double>::T, DOC(slp, VariableMatrix, T));
  cls.def("rows", &VariableMatrix<double>::rows,
          DOC(slp, VariableMatrix, rows));
  cls.def("cols", &VariableMatrix<double>::cols,
          DOC(slp, VariableMatrix, cols));
  cls.def_prop_ro("shape", [](const VariableMatrix<double>& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def("value",
          static_cast<double (VariableMatrix<double>::*)(int, int)>(
              &VariableMatrix<double>::value),
          "row"_a, "col"_a, DOC(slp, VariableMatrix, value));
  cls.def("value",
          static_cast<double (VariableMatrix<double>::*)(int)>(
              &VariableMatrix<double>::value),
          "index"_a, DOC(slp, VariableMatrix, value, 2));
  cls.def("value",
          static_cast<Eigen::MatrixXd (VariableMatrix<double>::*)()>(
              &VariableMatrix<double>::value),
          DOC(slp, VariableMatrix, value, 3));
  cls.def(
      "cwise_map",
      [](const VariableMatrix<double>& self,
         const std::function<Variable<double>(const Variable<double>& x)>&
             unary_op) { return self.cwise_transform(unary_op); },
      "func"_a, DOC(slp, VariableMatrix, cwise_transform));
  cls.def_static("zero", &VariableMatrix<double>::zero, "rows"_a, "cols"_a,
                 DOC(slp, VariableMatrix, zero));
  cls.def_static("ones", &VariableMatrix<double>::ones, "rows"_a, "cols"_a,
                 DOC(slp, VariableMatrix, ones));

  // Comparison operators
  cls.def(nb::self == nb::self, "rhs"_a, DOC(slp, operator, eq));
  cls.def(nb::self == Variable<double>(), "rhs"_a, DOC(slp, operator, eq));
  cls.def(nb::self == double(), "rhs"_a, DOC(slp, operator, eq));
  cls.def(nb::self == int(), "rhs"_a, DOC(slp, operator, eq));
  cls.def(Variable<double>() == nb::self, "lhs"_a, DOC(slp, operator, eq));
  cls.def(double() == nb::self, "lhs"_a, DOC(slp, operator, eq));
  cls.def(int() == nb::self, "lhs"_a, DOC(slp, operator, eq));
  cls.def(nb::self < nb::self, "rhs"_a, DOC(slp, operator, lt));
  cls.def(nb::self < Variable<double>(), "rhs"_a, DOC(slp, operator, lt));
  cls.def(nb::self < double(), "rhs"_a, DOC(slp, operator, lt));
  cls.def(nb::self < int(), "rhs"_a, DOC(slp, operator, lt));
  cls.def(Variable<double>() < nb::self, "lhs"_a, DOC(slp, operator, lt));
  cls.def(double() < nb::self, "lhs"_a, DOC(slp, operator, lt));
  cls.def(int() < nb::self, "lhs"_a, DOC(slp, operator, lt));
  cls.def(nb::self <= nb::self, "rhs"_a, DOC(slp, operator, le));
  cls.def(nb::self <= Variable<double>(), "rhs"_a, DOC(slp, operator, le));
  cls.def(nb::self <= double(), "rhs"_a, DOC(slp, operator, le));
  cls.def(nb::self <= int(), "rhs"_a, DOC(slp, operator, le));
  cls.def(Variable<double>() <= nb::self, "lhs"_a, DOC(slp, operator, le));
  cls.def(double() <= nb::self, "lhs"_a, DOC(slp, operator, le));
  cls.def(int() <= nb::self, "lhs"_a, DOC(slp, operator, le));
  cls.def(nb::self > nb::self, "rhs"_a, DOC(slp, operator, gt));
  cls.def(nb::self > Variable<double>(), "rhs"_a, DOC(slp, operator, gt));
  cls.def(nb::self > double(), "rhs"_a, DOC(slp, operator, gt));
  cls.def(nb::self > int(), "rhs"_a, DOC(slp, operator, gt));
  cls.def(Variable<double>() > nb::self, "lhs"_a, DOC(slp, operator, gt));
  cls.def(double() > nb::self, "lhs"_a, DOC(slp, operator, gt));
  cls.def(int() > nb::self, "lhs"_a, DOC(slp, operator, gt));
  cls.def(nb::self >= nb::self, "rhs"_a, DOC(slp, operator, ge));
  cls.def(nb::self >= Variable<double>(), "rhs"_a, DOC(slp, operator, ge));
  cls.def(nb::self >= double(), "rhs"_a, DOC(slp, operator, ge));
  cls.def(nb::self >= int(), "rhs"_a, DOC(slp, operator, ge));
  cls.def(Variable<double>() >= nb::self, "lhs"_a, DOC(slp, operator, ge));
  cls.def(double() >= nb::self, "lhs"_a, DOC(slp, operator, ge));
  cls.def(int() >= nb::self, "lhs"_a, DOC(slp, operator, ge));

  // VariableMatrix-VariableBlock overloads
  cls.def(
      "__mul__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableMatrix<double>& rhs,
         const VariableBlock<VariableMatrix<double>>& lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs + rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs - rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__eq__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs == rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq));
  cls.def(
      "__lt__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs < rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, lt));
  cls.def(
      "__le__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs <= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, le));
  cls.def(
      "__gt__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs > rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, gt));
  cls.def(
      "__ge__",
      [](const VariableMatrix<double>& lhs,
         const VariableBlock<VariableMatrix<double>>& rhs) {
        return lhs >= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, ge));
  cls.def(
      "__eq__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs == rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq));
  cls.def(
      "__lt__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs < rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, lt));
  cls.def(
      "__le__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs <= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, le));
  cls.def(
      "__gt__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs > rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, gt));
  cls.def(
      "__ge__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs >= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, ge));

  cls.def("__len__", &VariableMatrix<double>::rows,
          DOC(slp, VariableMatrix, rows));

  cls.def(
      "__iter__",
      [](const VariableMatrix<double>& self) {
        return nb::make_iterator(nb::type<VariableMatrix<double>>(),
                                 "value_iterator", self.begin(), self.end());
      },
      nb::keep_alive<0, 1>());

  autodiff.def(
      "cwise_reduce",
      [](const VariableMatrix<double>& lhs, const VariableMatrix<double>& rhs,
         const std::function<Variable<double>(const Variable<double>& x,
                                              const Variable<double>& y)>&
             binary_op) { return cwise_reduce<double>(lhs, rhs, binary_op); },
      "lhs"_a, "rhs"_a, "func"_a, DOC(slp, cwise_reduce));

  autodiff.def(
      "block",
      static_cast<VariableMatrix<double> (*)(
          const std::vector<std::vector<VariableMatrix<double>>>&)>(&block),
      "list"_a, DOC(slp, block));

  autodiff.def(
      "solve",
      static_cast<VariableMatrix<double> (*)(const VariableMatrix<double>&,
                                             const VariableMatrix<double>&)>(
          &solve),
      "A"_a, "B"_a, DOC(slp, solve));
}  // NOLINT(readability/fn_size)

}  // namespace slp
