// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <stdint.h>

#include <concepts>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <sleipnir/autodiff/slice.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/util/concepts.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "slice_tuple.hpp"
#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

void bind_variable_matrix(nb::module_& autodiff,
                          nb::class_<VariableMatrix<double>>& cls) {
  using namespace nb::literals;

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
      [](VariableMatrix<double>& self,
         nb::DRef<Eigen::MatrixX<int64_t>> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(slp, VariableMatrix, set_value));
  cls.def(
      "set_value",
      [](VariableMatrix<double>& self,
         nb::DRef<Eigen::MatrixX<int32_t>> values) {
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
      [](VariableMatrix<double>& self, SliceTuple slices, nb::object value) {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
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
      [](VariableMatrix<double>& self,
         std::tuple<int, int> slices) -> Variable<double>& {
        auto [row, col] = slices;

        if (row >= self.rows() || col >= self.cols()) {
          throw std::out_of_range("Index out of bounds");
        }

        if (row < 0) {
          row += self.rows();
        }
        if (col < 0) {
          col += self.cols();
        }
        return self[row, col];
      },
      nb::rv_policy::reference_internal, "slices"_a,
      DOC(slp, VariableMatrix, operator, array));
  cls.def(
      "__getitem__",
      [](VariableMatrix<double>& self,
         SliceTuple slices) -> VariableBlock<VariableMatrix<double>> {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
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

        return self[row_slice, row_slice_length, col_slice, col_slice_length];
      },
      nb::keep_alive<0, 1>(), "slices"_a,
      DOC(slp, VariableMatrix, operator, array));
  cls.def("row", nb::overload_cast<int>(&VariableMatrix<double>::row), "row"_a,
          DOC(slp, VariableMatrix, row));
  cls.def("col", nb::overload_cast<int>(&VariableMatrix<double>::col), "col"_a,
          DOC(slp, VariableMatrix, col));

  // Matrix-matrix multiplication. Products with a VariableBlock on the left are
  // bound on VariableBlock.
  for (const char* name : {"__mul__", "__matmul__"}) {
    cls.def(
        name,
        [](const VariableMatrix<double>& lhs,
           const VariableMatrix<double>& rhs) { return lhs * rhs; },
        nb::is_operator(), "rhs"_a);
    cls.def(
        name,
        [](const VariableMatrix<double>& lhs,
           const VariableBlock<VariableMatrix<double>>& rhs) {
          return lhs * rhs;
        },
        nb::is_operator(), "rhs"_a);
  }
  for (const char* name : {"__rmul__", "__rmatmul__"}) {
    cls.def(
        name,
        [](const VariableMatrix<double>& rhs,
           const VariableBlock<VariableMatrix<double>>& lhs) {
          return lhs * rhs;
        },
        nb::is_operator(), "lhs"_a);
  }

  // Matrix multiplication with NumPy arrays. These are also handled by
  // __array_ufunc__ below, but type checkers only see these overloads.
  cls.def(
      "__matmul__",
      [](const VariableMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmatmul__",
      [](const VariableMatrix<double>& rhs, nb::DRef<Eigen::MatrixXd> lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "lhs"_a);

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

        return nb::not_implemented();
      },
      // nanobind can't annotate nb::args or nb::kwargs, so the stub needs an
      // explicit signature
      nb::sig("def __array_ufunc__(self, ufunc: object, method: str, "
              "*inputs: object, **kwargs: object) -> object"),
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(-nb::self);

  // Matrix-scalar/scalar-matrix operations
  cls.def(nb::self * Variable<double>(), "rhs"_a);
  cls.def(Variable<double>() * nb::self, "lhs"_a);
  cls.def(nb::self / Variable<double>(), "rhs"_a);
  cls.def(nb::self /= Variable<double>(), "rhs"_a,
          DOC(slp, VariableMatrix, operator, idiv));
  for_each_type<double, int, const Variable<double>&,
                const VariableBlock<VariableMatrix<double>>,
                nb::DRef<Eigen::MatrixXd>>([&]<typename T> {
    cls.def(
        "__add__",
        [](const VariableMatrix<double>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs + Variable<double>{rhs};
          } else {
            return lhs + rhs;
          }
        },
        nb::is_operator(), "rhs"_a);
    cls.def(
        "__radd__",
        [](const VariableMatrix<double>& rhs, T&& lhs) {
          if constexpr (ScalarLike<T>) {
            return Variable<double>{lhs} + rhs;
          } else {
            return lhs + rhs;
          }
        },
        nb::is_operator(), "lhs"_a);
    cls.def(
        "__sub__",
        [](const VariableMatrix<double>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs - Variable<double>{rhs};
          } else {
            return lhs - rhs;
          }
        },
        nb::is_operator(), "rhs"_a);
    cls.def(
        "__rsub__",
        [](const VariableMatrix<double>& rhs, T&& lhs) {
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
      [](const VariableMatrix<double>& self, int power) {
        return self.cwise_transform(
            [=](const auto& elem) { return pow(elem, power); });
      },
      nb::is_operator(), "power"_a);
  cls.def_prop_ro("T", &VariableMatrix<double>::T, DOC(slp, VariableMatrix, T));
  cls.def("trace", &VariableMatrix<double>::trace,
          DOC(slp, VariableMatrix, trace));
  cls.def("rows", &VariableMatrix<double>::rows,
          DOC(slp, VariableMatrix, rows));
  cls.def("cols", &VariableMatrix<double>::cols,
          DOC(slp, VariableMatrix, cols));
  cls.def_prop_ro("shape", [](const VariableMatrix<double>& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def("value", nb::overload_cast<int, int>(&VariableMatrix<double>::value),
          "row"_a, "col"_a, DOC(slp, VariableMatrix, value));
  cls.def("value", nb::overload_cast<int>(&VariableMatrix<double>::value),
          "index"_a, DOC(slp, VariableMatrix, value, 2));
  cls.def("value", nb::overload_cast<>(&VariableMatrix<double>::value),
          DOC(slp, VariableMatrix, value, 3));
  cls.def(
      "cwise_map",
      [](const VariableMatrix<double>& self,
         const std::function<Variable<double>(const Variable<double>& x)>&
             unary_op) { return self.cwise_transform(unary_op); },
      "func"_a, DOC(slp, VariableMatrix, cwise_transform));
  cls.def("exp", &VariableMatrix<double>::exp, DOC(slp, VariableMatrix, exp));
  cls.def_static("identity", &VariableMatrix<double>::identity, "rows"_a,
                 DOC(slp, VariableMatrix, identity));
  cls.def_static("zero", &VariableMatrix<double>::zero, "rows"_a, "cols"_a,
                 DOC(slp, VariableMatrix, zero));
  cls.def_static("one", &VariableMatrix<double>::one, "rows"_a, "cols"_a,
                 DOC(slp, VariableMatrix, one));
  cls.def_static("constant", &VariableMatrix<double>::constant, "rows"_a,
                 "cols"_a, "constant"_a, DOC(slp, VariableMatrix, constant));

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
  for_each_type<const VariableBlock<VariableMatrix<double>>&,
                nb::DRef<Eigen::MatrixXd>>([&]<typename T> {
    cls.def(
        "__eq__",
        [](const VariableMatrix<double>& lhs, T&& rhs) { return lhs == rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, eq));
    cls.def(
        "__lt__",
        [](const VariableMatrix<double>& lhs, T&& rhs) { return lhs < rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, lt));
    cls.def(
        "__le__",
        [](const VariableMatrix<double>& lhs, T&& rhs) { return lhs <= rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, le));
    cls.def(
        "__gt__",
        [](const VariableMatrix<double>& lhs, T&& rhs) { return lhs > rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, gt));
    cls.def(
        "__ge__",
        [](const VariableMatrix<double>& lhs, T&& rhs) { return lhs >= rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, ge));
  });

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

  autodiff.def("block",
               nb::overload_cast<
                   const std::vector<std::vector<VariableMatrix<double>>>&>(
                   &block<double>),
               "list"_a, DOC(slp, block));

  autodiff.def("solve",
               nb::overload_cast<const VariableMatrix<double>&,
                                 const VariableMatrix<double>&>(&solve<double>),
               "A"_a, "B"_a, DOC(slp, solve));
  autodiff.def(
      "solve",
      [](const VariableMatrix<double>& A, nb::DRef<Eigen::MatrixXd> B) {
        return solve(A, VariableMatrix<double>(B));
      },
      "A"_a, "B"_a, DOC(slp, solve));
  autodiff.def(
      "solve",
      [](nb::DRef<Eigen::MatrixXd> A, const VariableMatrix<double>& B) {
        return solve(VariableMatrix<double>(A), B);
      },
      "A"_a, "B"_a, DOC(slp, solve));
}

}  // namespace slp
