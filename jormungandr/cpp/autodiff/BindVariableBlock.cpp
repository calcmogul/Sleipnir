// Copyright (c) Sleipnir contributors

#include <format>
#include <string>

#include <nanobind/eigen/dense.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>

#include "Docstrings.hpp"
#include "TryCast.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindVariableBlock(nb::class_<VariableBlock<VariableMatrix>>& cls) {
  using namespace nb::literals;

  // VariableBlock-VariableMatrix overloads
  cls.def(nb::self * VariableMatrix(), "rhs"_a);
  cls.def(nb::self + VariableMatrix(), "rhs"_a);
  cls.def(nb::self - VariableMatrix(), "rhs"_a);
  cls.def(nb::self == VariableMatrix(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self < VariableMatrix(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self <= VariableMatrix(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self > VariableMatrix(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self >= VariableMatrix(), "rhs"_a, DOC(sleipnir, operator, ge));

  cls.def(
      "set_value",
      [](VariableBlock<VariableMatrix>& self, double value) {
        self.SetValue(value);
      },
      "value"_a, DOC(sleipnir, VariableBlock, SetValue));
  cls.def(
      "set_value",
      [](VariableBlock<VariableMatrix>& self,
         nb::DRef<Eigen::MatrixXd> values) { self.SetValue(values); },
      "values"_a, DOC(sleipnir, VariableBlock, SetValue, 2));
  cls.def(
      "set_value",
      [](VariableBlock<VariableMatrix>& self,
         nb::DRef<Eigen::MatrixXf> values) {
        self.SetValue(values.cast<double>());
      },
      "values"_a, DOC(sleipnir, VariableBlock, SetValue, 2));
  cls.def(
      "set_value",
      [](VariableBlock<VariableMatrix>& self,
         nb::DRef<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>>
             values) { self.SetValue(values.cast<double>()); },
      "values"_a, DOC(sleipnir, VariableBlock, SetValue, 2));
  cls.def(
      "set_value",
      [](VariableBlock<VariableMatrix>& self,
         nb::DRef<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>>
             values) { self.SetValue(values.cast<double>()); },
      "values"_a, DOC(sleipnir, VariableBlock, SetValue, 2));
  cls.def(
      "__setitem__",
      [](VariableBlock<VariableMatrix>& self, int row, const Variable& value) {
        if (row < 0) {
          row += self.size();
        }
        return self[row] = value;
      },
      "row"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](VariableBlock<VariableMatrix>& self, nb::tuple slices,
         nb::object value) {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        Slice rowSlice;
        int rowSliceLength;
        Slice colSlice;
        int colSliceLength;

        // Row slice
        const auto& rowElem = slices[0];
        if (auto pyRowSlice = TryCast<nb::slice>(rowElem)) {
          auto t = pyRowSlice.value().compute(self.Rows());
          rowSlice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          rowSliceLength = t.get<3>();
        } else {
          int start = nb::cast<int>(rowElem);
          if (start < 0) {
            start += self.Rows();
          }
          rowSlice = Slice{start, start + 1};
          rowSliceLength = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (auto pyColSlice = TryCast<nb::slice>(colElem)) {
          auto t = pyColSlice.value().compute(self.Cols());
          colSlice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          colSliceLength = t.get<3>();
        } else {
          int start = nb::cast<int>(colElem);
          if (start < 0) {
            start += self.Cols();
          }
          colSlice = Slice{start, start + 1};
          colSliceLength = 1;
        }

        if (auto rhs = TryCast<VariableMatrix>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCast<VariableBlock<VariableMatrix>>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCastToEigen<double>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCastToEigen<float>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCastToEigen<int64_t>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCastToEigen<int32_t>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCast<Variable>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCast<double>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else if (auto rhs = TryCast<int>(value)) {
          self[rowSlice, rowSliceLength, colSlice, colSliceLength] =
              rhs.value();
        } else {
          throw nb::value_error(
              "VariableBlock.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.def(
      "__getitem__",
      [](VariableBlock<VariableMatrix>& self, int row) -> Variable& {
        if (row < 0) {
          row += self.size();
        }
        return self[row];
      },
      nb::keep_alive<0, 1>(), "row"_a,
      DOC(sleipnir, VariableBlock, operator, array, 3));
  cls.def(
      "__getitem__",
      [](VariableBlock<VariableMatrix>& self, nb::tuple slices) -> nb::object {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (nb::isinstance<int>(slices[0]) && nb::isinstance<int>(slices[1])) {
          int row = nb::cast<int>(slices[0]);
          int col = nb::cast<int>(slices[1]);

          if (row >= self.Rows() || col >= self.Cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row += self.Rows();
          }
          if (col < 0) {
            col += self.Cols();
          }
          return nb::cast(self[row, col]);
        }

        Slice rowSlice;
        int rowSliceLength;
        Slice colSlice;
        int colSliceLength;

        // Row slice
        const auto& rowElem = slices[0];
        if (auto pyRowSlice = TryCast<nb::slice>(rowElem)) {
          auto t = pyRowSlice.value().compute(self.Rows());
          rowSlice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          rowSliceLength = t.get<3>();
        } else {
          int start = nb::cast<int>(rowElem);
          if (start < 0) {
            start += self.Rows();
          }
          rowSlice = Slice{start, start + 1};
          rowSliceLength = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (auto pyColSlice = TryCast<nb::slice>(colElem)) {
          auto t = pyColSlice.value().compute(self.Cols());
          colSlice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          colSliceLength = t.get<3>();
        } else {
          int start = nb::cast<int>(colElem);
          if (start < 0) {
            start += self.Cols();
          }
          colSlice = Slice{start, start + 1};
          colSliceLength = 1;
        }

        return nb::cast(
            self[rowSlice, rowSliceLength, colSlice, colSliceLength]);
      },
      nb::keep_alive<0, 1>(), DOC(sleipnir, VariableBlock, operator, array),
      "slices"_a);
  cls.def("row", nb::overload_cast<int>(&VariableBlock<VariableMatrix>::Row),
          "row"_a, DOC(sleipnir, VariableBlock, Row));
  cls.def("col", nb::overload_cast<int>(&VariableBlock<VariableMatrix>::Col),
          "col"_a, DOC(sleipnir, VariableBlock, Col));
  cls.def(
      "__mul__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const VariableBlock<VariableMatrix>& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableBlock<VariableMatrix>& rhs,
         const VariableBlock<VariableMatrix>& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const VariableBlock<VariableMatrix>& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a);

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](VariableBlock<VariableMatrix>& self, nb::object ufunc, nb::str method,
         nb::args inputs, const nb::kwargs&) -> nb::object {
        std::string method_name = nb::cast<std::string>(method);
        std::string ufunc_name =
            nb::cast<std::string>(ufunc.attr("__repr__")());

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = TryCastToEigen<float>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = TryCastToEigen<float>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = TryCastToEigen<float>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = TryCastToEigen<float>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = TryCastToEigen<float>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = TryCastToEigen<float>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = TryCastToEigen<float>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = TryCastToEigen<float>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            }
          }
        }

        std::string input1_name =
            nb::cast<std::string>(inputs[0].attr("__repr__")());
        std::string input2_name =
            nb::cast<std::string>(inputs[1].attr("__repr__")());
        throw nb::value_error(
            std::format("VariableBlock: numpy method {}, ufunc {} not "
                        "implemented for ({}, {})",
                        method_name, ufunc_name, input1_name, input2_name)
                .c_str());
        return nb::cast(VariableMatrix{self});
      },
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  cls.def(nb::self * Variable(), "rhs"_a);
  cls.def(nb::self * double(), "rhs"_a);
  cls.def(Variable() * nb::self, "lhs"_a);
  cls.def(double() * nb::self, "lhs"_a);
  cls.def(nb::self / Variable(), "rhs"_a);
  cls.def(nb::self / double(), "rhs"_a);
  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(
      "__add__",
      [](const VariableBlock<VariableMatrix>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) -> VariableMatrix { return lhs + rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__add__",
      [](nb::DRef<Eigen::MatrixXd> lhs,
         const VariableBlock<VariableMatrix>& rhs) -> VariableMatrix {
        return lhs + rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(
      "__sub__",
      [](const VariableBlock<VariableMatrix>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) -> VariableMatrix { return lhs - rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__sub__",
      [](nb::DRef<Eigen::MatrixXd> lhs,
         const VariableBlock<VariableMatrix>& rhs) -> VariableMatrix {
        return lhs - rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(-nb::self);
  cls.def(
      "__pow__",
      [](const VariableBlock<VariableMatrix>& self, int power) {
        return sleipnir::pow(VariableMatrix{self}, power);
      },
      nb::is_operator(), "power"_a);
  cls.def_prop_ro("T", &VariableBlock<VariableMatrix>::T,
                  DOC(sleipnir, VariableBlock, T));
  cls.def("rows", &VariableBlock<VariableMatrix>::Rows,
          DOC(sleipnir, VariableBlock, Rows));
  cls.def("cols", &VariableBlock<VariableMatrix>::Cols,
          DOC(sleipnir, VariableBlock, Cols));
  cls.def_prop_ro("shape", [](const VariableBlock<VariableMatrix>& self) {
    return nb::make_tuple(self.Rows(), self.Cols());
  });
  cls.def("value",
          static_cast<double (VariableBlock<VariableMatrix>::*)(int, int)>(
              &VariableBlock<VariableMatrix>::Value),
          "row"_a, "col"_a, DOC(sleipnir, VariableBlock, Value));
  cls.def("value",
          static_cast<double (VariableBlock<VariableMatrix>::*)(int)>(
              &VariableBlock<VariableMatrix>::Value),
          "index"_a, DOC(sleipnir, VariableBlock, Value, 2));
  cls.def("value",
          static_cast<Eigen::MatrixXd (VariableBlock<VariableMatrix>::*)()>(
              &VariableBlock<VariableMatrix>::Value),
          DOC(sleipnir, VariableBlock, Value, 3));
  cls.def(
      "cwise_transform",
      [](const VariableBlock<VariableMatrix>& self,
         const std::function<Variable(const Variable& x)>& unaryOp) {
        return self.CwiseTransform(unaryOp);
      },
      "func"_a, DOC(sleipnir, VariableBlock, CwiseTransform));
  cls.def(nb::self == nb::self, "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self == Variable(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self == double(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self == int(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(Variable() == nb::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(double() == nb::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(int() == nb::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self < nb::self, "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self < Variable(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self < double(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self < int(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(Variable() < nb::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(double() < nb::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(int() < nb::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self <= nb::self, "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self <= Variable(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self <= double(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self <= int(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(Variable() <= nb::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(double() <= nb::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(int() <= nb::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self > nb::self, "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self > Variable(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self > double(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self > int(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(Variable() > nb::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(double() > nb::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(int() > nb::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self >= nb::self, "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(nb::self >= Variable(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(nb::self >= double(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(nb::self >= int(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(Variable() >= nb::self, "lhs"_a, DOC(sleipnir, operator, ge));
  cls.def(double() >= nb::self, "lhs"_a, DOC(sleipnir, operator, ge));
  cls.def(int() >= nb::self, "lhs"_a, DOC(sleipnir, operator, ge));
  cls.def(
      "__eq__",
      [](const VariableBlock<VariableMatrix>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(
      "__lt__",
      [](const VariableBlock<VariableMatrix>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) { return lhs < rhs; },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(
      "__le__",
      [](const VariableBlock<VariableMatrix>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) { return lhs <= rhs; },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(
      "__gt__",
      [](const VariableBlock<VariableMatrix>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) { return lhs > rhs; },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(
      "__ge__",
      [](const VariableBlock<VariableMatrix>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) { return lhs >= rhs; },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, ge));

  cls.def("__len__", &VariableBlock<VariableMatrix>::Rows,
          DOC(sleipnir, VariableBlock, Rows));

  cls.def(
      "__iter__",
      [](const VariableBlock<VariableMatrix>& self) {
        return nb::make_iterator(nb::type<VariableBlock<VariableMatrix>>(),
                                 "value_iterator", self.begin(), self.end());
      },
      nb::keep_alive<0, 1>());
}  // NOLINT(readability/fn_size)

}  // namespace sleipnir
